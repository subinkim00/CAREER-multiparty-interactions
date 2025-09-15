import os
import re
import webbrowser
import pandas as pd
from boxsdk import OAuth2, Client
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import time

#################################################################################
# RUN LOCAL SERVER FOR 2 FACTOR AUTHENTICATION
#################################################################################
class OAuthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # suppress server logs
        self.log_message = lambda format, *args: None
        
        query = urlparse(self.path).query
        params = parse_qs(query)
        self.server.auth_code = params.get('code', [None])[0]
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'Authentication successful. You can close this window.')
        
        # signal the server to shutdown
        self.server.shutdown_requested = True

#################################################################################
# FETCH AUTHENTICATION CODE
#################################################################################
def get_auth_code(client_id, redirect_uri, port=8080):
    auth_url = (
        f"https://account.box.com/api/oauth2/authorize?"
        f"response_type=code&client_id={client_id}&redirect_uri={redirect_uri}"
        f"&scope=root_readwrite"
    )
    
    # update redirect_uri to include the port
    redirect_with_port = f"{redirect_uri}:{port}"
    auth_url = auth_url.replace(redirect_uri, redirect_with_port)
    
    print(f"Opening browser to authorize: {auth_url}")
    webbrowser.open(auth_url)
    
    server_address = ('localhost', port)
    server = HTTPServer(server_address, OAuthHandler)
    server.auth_code = None
    server.shutdown_requested = False
    
    # run server in interruptable thread
    def server_thread():
        while not server.shutdown_requested:
            server.handle_request()
    
    thread = threading.Thread(target=server_thread)
    thread.daemon = True
    thread.start()
    
    # wait for auth code with timeout
    timeout = 300  # 5 mins timeout
    start_time = time.time()
    while server.auth_code is None:
        if time.time() - start_time > timeout:
            print("Timeout waiting for authentication")
            return None
        time.sleep(0.5)
    
    # give the server time to send the response
    time.sleep(1)
    
    return server.auth_code

#################################################################################
# AUTHENTICATING ON BROWSER
# using info from box developer interface
#################################################################################

CLIENT_ID = os.environ.get("BOX_CLIENT_ID")
CLIENT_SECRET = os.environ.get("BOX_CLIENT_SECRET")

# hardcoded values as fallback for environment variables
if not CLIENT_ID:
    CLIENT_ID = "vdxa3xbitg99n9oi6fwjgdnz7d152omd"
if not CLIENT_SECRET:
    CLIENT_SECRET = "5W12sHkJZP2mKtkBebLk4h2HlHlMsvsR"

REDIRECT_URI = "http://localhost"
PORT = 8080

print("🔑 Opening browser for Box login...")
auth_code = get_auth_code(CLIENT_ID, REDIRECT_URI, PORT)

if not auth_code:
    print("❌ Failed to get authentication code. Exiting.")
    exit(1)

oauth = OAuth2(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
)

try:
    access_token, refresh_token = oauth.authenticate(auth_code)
    client = Client(oauth)
    print("✅ Authenticated with Box!")
except Exception as e:
    print(f"❌ Authentication failed: {e}")
    exit(1)

#################################################################################
# LOADING SPREADSHEET
# contains multispeaker recordings identified with identify_multiparty_recs.py
#################################################################################
try:
    df = pd.read_excel("multispeaker_output.xlsx")
    print(f"📊 Loaded spreadsheet with {len(df)} rows")
except Exception as e:
    print(f"❌ Failed to load spreadsheet: {e}")
    exit(1)

#################################################################################
# SUBFOLDER LOGIC
# if file has been supercoded, want .eaf file from completed/XX/checks
# else, completed/XX where XX is initials of coder/supercoder
# super coder is under "Super coder" col, annotator is under "Annotator" col
#################################################################################
def determine_subfolder(row):
    super_coder = row.get("Super coder", "")
    annotator = row.get("Annotator", "")
    if pd.notna(super_coder) and re.search(r'[A-Za-z]', str(super_coder)):
        return f"{str(super_coder).strip()}/checks"
    elif pd.notna(annotator):
        return str(annotator).strip()
    else:
        return None

#################################################################################
# FIND BASE BOX DIRECTORY
#################################################################################
def find_base_directory(client):
    path_parts = ["ChatterLab", "Member work directories", "CAREER transcription team", "annotator_files", "completed"]
    current = client.folder(folder_id='0')  # root
    
    print("🔍 Finding base directory...")
    for part in path_parts:
        found = False
        try:
            items = list(current.get_items(limit=1000))
            for item in items:
                if item.type == 'folder':
                    item_info = item.get()
                    if item_info.name == part:
                        current = item
                        print(f"  ✓ Found: {part}")
                        found = True
                        break
            
            if not found:
                print(f"❌ Could not find '{part}' subfolder. Please check the path.")
                return None
                
        except Exception as e:
            print(f"❌ Error navigating to base directory: {str(e)}")
            return None
    
    print(f"✅ Found base directory: {path_parts[-1]}")
    return current

#################################################################################
# TRAVERSE GIVEN DIRECTORY TO LOCATE FILE
#################################################################################
def get_folder_by_path(path_parts, base_folder):
    current = base_folder
    
    for part in path_parts:
        if not part:  # skip empty parts
            continue
            
        found = False
        try:
            items = list(current.get_items(limit=1000))
            
            for item in items:
                if item.type == 'folder':
                    item_info = item.get()
                    if item_info.name == part:
                        current = item
                        found = True
                        break
                        
            if not found:
                print(f"❌ Subfolder '{part}' not found in '{current.get().name}'")
                return None
                
        except Exception as e:
            print(f"❌ Error accessing folder: {str(e)}")
            return None
            
    return current

# download loop
download_dir = "downloaded_eafs"
os.makedirs(download_dir, exist_ok=True)
print(f"📁 Files will be downloaded to: {os.path.abspath(download_dir)}")

# find the base directory once
base_folder = find_base_directory(client)
if not base_folder:
    print("❌ Could not find the base directory. Exiting.")
    exit(1)

success_count = 0
failed_count = 0

for idx, row in df.iterrows():
    rec_name = str(row.get("Recording", "")).strip()
    folder_path = determine_subfolder(row)

    if not rec_name or not folder_path:
        print(f"⚠️ Skipping row {idx+1} with missing info: {row}")
        failed_count += 1
        continue

    target_file = f"{rec_name}.eaf"
    sarah_file = f"{rec_name}_SS.eaf" # naming discrepancies :')
    path_parts = folder_path.split("/")

    print(f"\n🔍 [{idx+1}/{len(df)}] Looking for {target_file} in folder: {folder_path}")
    folder = get_folder_by_path(path_parts, base_folder)

    if not folder:
        print(f"❌ Folder {folder_path} not found in Box.")
        failed_count += 1
        continue

    found_file = None
    for item in folder.get_items(limit=1000):
        if item.type == 'file' and (item.name == target_file or item.name == sarah_file):
            found_file = item
            break

    if not found_file:
        print(f"❌ {target_file} not found in {folder_path}")
        failed_count += 1
        continue

    try:
        local_path = os.path.join(download_dir, target_file)
        with open(local_path, 'wb') as f:
            found_file.download_to(f)
        print(f"✅ Downloaded: {target_file}")
        success_count += 1
    except Exception as e:
        print(f"❌ Failed to download {target_file}: {e}")
        failed_count += 1

print(f"\n📊 Summary: {success_count} files downloaded, {failed_count} failed")