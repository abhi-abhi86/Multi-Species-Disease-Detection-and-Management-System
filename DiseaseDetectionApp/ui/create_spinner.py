import base64
import os

                                                                            
GIF_BASE64_DATA = """
R0lGODlhIAAgAPMAAP///wAAAMLCwkJCQgAAAGJiYoKCgpKSkiH/C05FVFNDQVBFMi4wAwEAAAAh
/hpDcmVhdGVkIHdpdGggYWpheGxvYWQuaW5mbwAh+QQJCgAAACwAAAAAIAAgAAAE5xDISWlZ
Go2bC2iE3kyOhmM4lD2iS3G0fJ8LmhxoVnVa26sI7TjD27w8scx/I4kI4CRNGiQksg3s0xM0
oAFiS0Z3RzUfDxQHkQ4TQ5RweK4hSa02t3QBAM42f0B459gaGhoZ2p8f258YnB0Yn5/Z2N1
f3p8f3p9Z2l7Z3l5f2l5Z2N8Y3t/Z3B4fXp4e3t/eXp8f3p9Z2l8Z3p8Z3p9Z2l7Z3l5f2l5
Z2N8Y3t/Z3B4fXp4e3t/eXp8f3p9Z2l8Z3p8Z3p9Z2l7Z3l5f2l5Z2N8Y3t/Z3B4fXp4e3t/
eXp8f3p9Z2l7Z3l5f2l5Z2N8Y3t/Z3B4fXp4e3t/eXp8f3p9Z2l8Z3p8Z3p9Z2l7Z3l5f2l5
Z2N8Y3t/Z3B4fXp4e3t/eXp8f3p9Z2l8Z3p8Z3p9Z2l7Z3l5f2l5Z2N8Y3t/Z3B4fXp4e3t/
eXp8f3p9Z2l7Z3l5f2l5Z2N8Y3t/Z3B4fXp4e3t/eXp8f3p9Z2l8Z3p8Z3p9Z2l7Z3l5f2l5
Z2N8Y3t/Z3B4fXp4e3t/eXp8f3p9Z2l8Z3p8Z3p9Z2l7Z3l5f2l5Z2N8Y3t/Z3B4fXp4e3t/
eXp8f3p9Z2l8Z3p8Z3p9Z2l7Z3l5f2l5Z2N8Y3t/Z3B4fXp4e3t/eXp8f3p9Z2l7Z3l5f2l5
Z2N8Y3t/Z3B4fXp4e3t/eXp8f3p9Z2l8Z3p8Z3p9Z2l7Z3l5f2l5Z2N8Y3t/Z3B4fXp4e3t/
eXp8f3p9Z2l8Z3p8Z3p9Z2l7Z3l5f2l5Z2N8Y3t/Z3B4fXp4e3t/eXp8f3p9Z2l7Z3l5f2l5
Z2N8Y3t/Z3B4eXhwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP4dTE1VzI0b0E5a2t3
eGFsN3JzZnlxN3V5N3d4eHh3eHh3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3
d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3
d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3
d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3
d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3
d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3
d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3
d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3
d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3
d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3
d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3
d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3
d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3
d3d3d3d3d3d3d3d3d//Rw0KGgoAAAANSUhEUgAAACAA
AAAgCAYAAABzenr0AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYA
AICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAZiS0dEAAAAAAAA+UO7fwAAAAlw
SFlzAAAASAAAAEgARslrPgAAACF6VFh0Q29tbWVudAAyMDE5LTAyLTE4VDAxOjA5OjU3LTA3
OjAwIG1zYW11ZWxzj19DAAADVklEQVRYCe1WW2+bRhA+Z/9/8L/hG/5RfYhsrWq3u93t7u3u
3d3d7e7e3e3u3t3t7t7d7e3u3t3t7t7d7e3u3t3t7t7d7e3u3t3t7t7d7e3u3t3t7t7d7e3u
3t3t7t7d7e3u3t3t7t7d7e3u3t3t7t7d7e3u3t3t7t7d7e3u3t3t7t7d7e3u3t3t7t7d7e3u
3t7d7e3u3t3t7t7d7e3u3t3t7t7d7e3t3t7t7t7d7e3t3t7t7d7e3t3t7t7d7e3t3t7t7d7t
7t7d7e3u3t7t7d7e3u3t3t7d7e3u3t3t7t7d7e3u3t7d7e3u3t7d7e3u3t7d7e3t3t7t7d7t
7t7d7e3t3t7d7e3u3t7d7e3u3t7d7e3t3t7t7t7d7e3t3t7d7e3t7t7d7e3u3t7d7e3u3t7t
7t7d7e3t7t7d7e3t7t7d7e3t7t7d7e3t7t7d7e3t3t7t7d7e3t7t7t7d7e3t7t7d7e3t7t7t
7t7d7e3t7t7d7e3u3t7t7d7e3u3t7t7d7e3u3t7d7e3u3t7d7e3t7t7d7e3t7t7d7e3u3t3t
7t7d7e3t7t7d7e3u3t7t7t7d7e3t7t7d7e3t7t7d7e3u3t7d7e3t7t7d7e3t7t7t7d7e3t7t
7t7d7e3t7t7d7e3t7t7d7e3t7t7d7e3t7t7d7e3t7t7d7e3u3t7d7e3t7t7d7e3t7t7d7e3t
7t7d7e3t7t7d7e3u3t7d7e3u3t7d7e3t7t7d7e3u3t7d7e3u3t7d7e3t7t7d7e3t7t7d7e3t
7t7d7e3t7t7d7e3t7t7d7e3t7t7d7e3t7t7d7e3t7t7d7e3t7t7d7e3t7t7d7e3t7t7d7e3t
7t7d7e3t7t7d7e3t7t7d7e3t7t7d7e3t7t7d7e3t7t7d7e3t7t7d7e3u3t7t7t7d7e3t7t7t
7t7d7e3t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
7t7t7d7e3t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t7t
t-"""

def create_gif():
    """Decodes the base64 string and writes the spinner.gif file."""
                                                                    
                                                           
                                                               
    output_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, 'spinner.gif')

    try:
                                                               
        gif_data = base64.b64decode(GIF_BASE64_DATA)
        
                                           
        with open(file_path, 'wb') as f:
            f.write(gif_data)
        
        print(f"Successfully created 'spinner.gif' at: {file_path}")

    except Exception as e:
        print(f"An error occurred while creating the GIF: {e}")

if __name__ == "__main__":
    create_gif()
