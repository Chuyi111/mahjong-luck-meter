import asyncio
import hashlib
import hmac
import logging
import os
import random
import uuid
from argparse import ArgumentParser
from pathlib import Path

import aiohttp

from ms.base import MSRPCChannel
from ms.rpc import Lobby
import ms.protocol_pb2 as pb
from google.protobuf.json_format import MessageToJson


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

MS_HOST = "https://game.maj-soul.com"


async def main():
    """
    Log in to Mahjong Soul (CN server), fetch recent game UUIDs,
    download each game record, and save as JSON files.

    Usage example:
        python download_logs.py -u USER -p PASS --limit 50 --output-dir logs
        python download_logs.py -u USER -p PASS --uuid SOME-RECORD-UUID
    """
async def main():
    """
    Log in to Mahjong Soul (CN server), fetch recent game UUIDs,
    download each game record, and save as JSON files.
    """
    parser = ArgumentParser(description="Download Mahjong Soul game logs (CN server).")
    parser.add_argument("-u", "--username", type=str, required=True,
                        help="Mahjong Soul account name (CN server).")
    parser.add_argument("-p", "--password", type=str, required=True,
                        help="Mahjong Soul account password (CN server).")
    parser.add_argument("-l", "--limit", type=int, default=30,
                        help="Number of latest logs to fetch (default: 30). Ignored if --uuid is given.")
    parser.add_argument("--uuid", type=str, default=None,
                        help="Download a single game log by its UUID instead of a batch.")
    parser.add_argument("-o", "--output-dir", type=str, default="logs",
                        help="Directory to save downloaded logs (default: ./logs).")
    parser.add_argument(
        "--ws-endpoint",
        type=str,
        default=None,
        help="Override websocket endpoint, e.g. wss://gateway-xxxx.maj-soul.com/gateway",
    )

    opts = parser.parse_args()  
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lobby, channel, version_to_force = await connect(opts.ws_endpoint)


    ok = await login(lobby, opts.username, opts.password, version_to_force)
    if not ok:
        logging.error("Login failed, aborting.")
        await channel.close()
        return

    if opts.uuid:
        # Download a single specified log
        await download_single_log(
            lobby=lobby,
            uuid_str=opts.uuid,
            version_to_force=version_to_force,
            output_dir=output_dir,
        )
    else:
        # Download a batch of most recent logs
        uuids = await load_game_logs(lobby, limit=opts.limit)
        logging.info("Found %d record UUIDs", len(uuids))

        for i, game_uuid in enumerate(uuids, start=1):
            logging.info("Downloading log %d/%d: %s", i, len(uuids), game_uuid)
            await download_single_log(
                lobby=lobby,
                uuid_str=game_uuid,
                version_to_force=version_to_force,
                output_dir=output_dir,
            )

    await channel.close()
    logging.info("Done.")



async def connect(ws_endpoint: str | None):
    """
    Get Mahjong Soul version info and open an MSRPCChannel.

    If ws_endpoint is provided, we skip automatic discovery and just
    connect directly to that websocket URL.

    If ws_endpoint is None, we *try* automatic discovery via config['ip'],
    but if the route returns HTML (not JSON), we raise with a clear message.
    """
    async with aiohttp.ClientSession() as session:
        # Always get version.json so we know the correct client_version_string
        async with session.get(f"{MS_HOST}/1/version.json") as res:
            version_info = await res.json()
            logging.info("Version JSON: %s", version_info)
            version = version_info["version"]
            version_to_force = version.replace(".w", "")

        # If user supplied a websocket endpoint, just use it
        if ws_endpoint:
            endpoint = ws_endpoint
            logging.info("Using user-specified websocket endpoint: %s", endpoint)
            channel = MSRPCChannel(endpoint)
            lobby = Lobby(channel)
            await channel.connect(MS_HOST)
            logging.info("Connection established (manual endpoint).")
            return lobby, channel, version_to_force

        # Otherwise, try automatic discovery (may not work if server changed)
        async with session.get(f"{MS_HOST}/1/v{version}/config.json") as res:
            config = await res.json()
            logging.info("Config JSON: %s", config)

        ip_list = config.get("ip")
        if not ip_list or not isinstance(ip_list, list):
            raise RuntimeError("config['ip'] is missing or not a list")

        ip_entry = ip_list[0]
        logging.info("Using ip entry: %r", ip_entry)

        gateways = ip_entry.get("gateways")
        if not gateways or not isinstance(gateways, list):
            raise RuntimeError("config['ip'][0] has no 'gateways' list")

        gw = random.choice(gateways)
        base_url = gw.get("url")
        if not base_url:
            raise RuntimeError("Chosen gateway entry has no 'url' field")

        logging.info("Chosen HTTP route for gateway discovery: %s", base_url)

        discovery_url = base_url + "?service=ws-gateway&protocol=ws&ssl=true"
        async with session.get(discovery_url) as res:
            content_type = res.headers.get("Content-Type", "")
            text = await res.text()
            if "application/json" not in content_type:
                logging.error(
                    "Gateway discovery URL returned non-JSON content-type (%s). "
                    "This likely means the route now returns HTML (e.g., an error page) "
                    "instead of the JSON server list.\n"
                    "Response snippet: %.200s",
                    content_type,
                    text,
                )
                raise RuntimeError(
                    "Automatic gateway discovery failed (HTML instead of JSON).\n"
                    "Please get the working websocket URL from your browser's DevTools "
                    "and pass it via --ws-endpoint."
                )

            servers_info = await res.json()
            logging.info("Available servers from %s: %s", discovery_url, servers_info)
            servers = servers_info.get("servers")
            if not servers:
                raise RuntimeError("No 'servers' field in gateway discovery response")

    # Build websocket endpoint(s) from 'servers' (fallback path, probably unused once you use --ws-endpoint)
    candidate_endpoints = []
    for s in servers:
        if isinstance(s, str):
            addr = s
        elif isinstance(s, dict):
            addr = (
                s.get("url")
                or s.get("addr")
                or s.get("endpoint")
                or s.get("host")
                or s.get("server")
            )
            if not isinstance(addr, str):
                continue
        else:
            continue

        if addr.startswith("ws://") or addr.startswith("wss://"):
            endpoint = addr
        else:
            endpoint = f"wss://{addr}/gateway"

        candidate_endpoints.append(endpoint)

    candidate_endpoints = list(dict.fromkeys(candidate_endpoints))  # dedupe

    if not candidate_endpoints:
        raise RuntimeError("No websocket endpoints could be constructed from 'servers' list.")

    last_error = None
    for endpoint in candidate_endpoints:
        logging.info("Trying websocket endpoint: %s", endpoint)
        channel = MSRPCChannel(endpoint)
        lobby = Lobby(channel)
        try:
            await channel.connect(MS_HOST)
            logging.info("Connection established via %s", endpoint)
            return lobby, channel, version_to_force
        except Exception as e:
            logging.warning("Failed to connect to %s: %s", endpoint, e)
            last_error = e
            try:
                await channel.close()
            except Exception:
                pass

    raise RuntimeError(f"Failed to connect to any websocket endpoint. Last error: {last_error}")


async def login(lobby, username, password, version_to_force):
    """
    Login using the same mechanism as example.py (CN account + password).
    Returns True on success, False otherwise.
    """
    logging.info("Logging in with username and password.")

    uuid_key = str(uuid.uuid1())

    req = pb.ReqLogin()
    req.account = username
    # Same HMAC salt used by web client / example.py
    req.password = hmac.new(b"lailai", password.encode(), hashlib.sha256).hexdigest()
    req.device.is_browser = True
    req.random_key = uuid_key
    req.gen_access_token = True
    req.client_version_string = f"web-{version_to_force}"
    req.currency_platforms.append(2)

    res = await lobby.login(req)

    token = res.access_token
    if not token:
        logging.error("Login error: %s", res)
        return False

    logging.info("Login succeeded.")
    return True


async def load_game_logs(lobby, limit=30):
    """
    Fetch a list of recent game record UUIDs.
    This mirrors example.py's fetch_game_record_list usage,
    but allows a configurable count (up to 'limit').
    """
    logging.info("Loading up to %d game log UUIDs.", limit)

    req = pb.ReqGameRecordList()
    req.start = 1           # 1-based index; latest records
    req.count = limit

    res = await lobby.fetch_game_record_list(req)
    uuids = [r.uuid for r in res.record_list]

    logging.info("Server returned %d records.", len(uuids))
    return uuids


async def download_single_log(lobby, uuid_str, version_to_force, output_dir: Path):
    """
    Fetch a single game record (by UUID), parse GameDetailRecords, and
    save it as JSON to output_dir/<uuid>.json
    """
    logging.info("Fetching game log %s", uuid_str)

    req = pb.ReqGameRecord()
    req.game_uuid = uuid_str
    req.client_version_string = f"web-{version_to_force}"

    try:
        res = await lobby.fetch_game_record(req)
    except Exception as e:
        logging.error("Failed to fetch game %s: %s", uuid_str, e)
        return

    # res.data is a Wrapper around GameDetailRecords
    record_wrapper = pb.Wrapper()
    record_wrapper.ParseFromString(res.data)

    game_details = pb.GameDetailRecords()
    game_details.ParseFromString(record_wrapper.data)

    # Convert the full GameDetailRecords proto into JSON
    json_str = MessageToJson(game_details)

    # Write JSON to disk
    out_path = output_dir / f"{uuid_str}.json"
    try:
        with out_path.open("w", encoding="utf-8") as f:
            f.write(json_str)
        logging.info("Saved game %s to %s", uuid_str, out_path)
    except OSError as e:
        logging.error("Failed to write file %s: %s", out_path, e)


if __name__ == "__main__":
    asyncio.run(main())
