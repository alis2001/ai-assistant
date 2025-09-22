import asyncio
import websockets
import pyaudio
import uuid
import time

clients = set()  # To store the connected clients

async def handle_audio_stream(websocket, path):
    session_uuid = str(uuid.uuid4())  # Unique identifier for each connection
    clients.add(websocket)  # Add the new client to the clients set
    print(f"New connection: {session_uuid}, Path: {path}")
    
    try:
        while True:
            # Receive audio data from the client
            audio_data = await websocket.recv()
            print(f"Received audio chunk of size: {len(audio_data)} bytes.")

            # Broadcast the received audio to all clients
            for client in clients:
                if client != websocket:  # Don't send back to the sender
                    await client.send(audio_data)
                    print(f"Sent audio to client: {client}")
                    
    except websockets.ConnectionClosed:
        print(f"Connection closed: {session_uuid}")
    finally:
        clients.remove(websocket)  # Remove the client from the list when disconnected

# WebSocket server
async def main():
    server = await websockets.serve(handle_audio_stream, "0.0.0.0", 7888)
    print("WebSocket server running on ws://0.0.0.0:7888")
    await server.wait_closed()

# Run the server
asyncio.run(main())



