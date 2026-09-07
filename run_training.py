import subprocess
import time
from nn_client import Client

for i in range(300):
    process = subprocess.Popen([
        "rcssservermj",
        "--host", "127.0.0.1",
        "--aport", "60000",
        "--mport", "60001",
        "--field", "hsl_m_26"
    ])
    print(f"Server started in background with PID: {process.pid}")

    time.sleep(5)
    # TODO: change the initial position everytime
    client = Client(
        host='127.0.0.1',
        port=60000,
        team='Test',
        player_no=1,
        model_name='T1',
    )
    client.run()
    print(client.elapsed_time) # TODO write this into a file
    time.sleep(5)
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("Server did not stop in time. Forcing kill...")
        process.kill()
        process.wait()
        
    print("Server stopped.")