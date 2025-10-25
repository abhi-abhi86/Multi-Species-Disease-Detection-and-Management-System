



import json
import os
import sys
from pathlib import Path

def get_developer_info():
    """
    Access developer information - only available to authorized developer.
    """

    project_root = Path(__file__).parent


    info_file = project_root / '.developer_info.json'


    if not info_file.exists():
        print("❌ Developer information file not found.")
        return False


    username = input("Enter developer username: ").strip()


    if username != "abhi-abhi86":
        print("❌ Access denied. Invalid username.")
        return False

    try:

        with open(info_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print("\n" + "="*60)
        print("👨‍💻 DEVELOPER INFORMATION")
        print("="*60)

        developer = data.get('developer', {})
        print(f"Name: {developer.get('name', 'N/A')}")
        print(f"Username: {developer.get('username', 'N/A')}")
        print(f"GitHub: {developer.get('github', 'N/A')}")
        print(f"Email: {developer.get('email', 'N/A')}")
        print(f"Project: {developer.get('project', 'N/A')}")
        print(f"Version: {developer.get('version', 'N/A')}")
        print(f"Description: {developer.get('description', 'N/A')}")

        print(f"\nTechnologies: {', '.join(developer.get('technologies', []))}")
        print(f"Features: {', '.join(developer.get('features', []))}")

        system_info = data.get('system_info', {})
        print(f"\nSystem Requirements:")
        print(f"  OS: {system_info.get('os', 'N/A')}")
        print(f"  Python: {system_info.get('python_version', 'N/A')}")
        print(f"  Dependencies: {', '.join(system_info.get('dependencies', []))}")

        print("\n" + "="*60)
        print("✅ Access granted. Developer information displayed.")
        return True

    except json.JSONDecodeError:
        print("❌ Error: Invalid developer information file format.")
        return False
    except Exception as e:
        print(f"❌ Error accessing developer information: {e}")
        return False

if __name__ == "__main__":
    print("🔐 Developer Information Access")
    print("This information is only accessible to the authorized developer.")
    print("-" * 50)

    success = get_developer_info()
    if not success:
        sys.exit(1)
