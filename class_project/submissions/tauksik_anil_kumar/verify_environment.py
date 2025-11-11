import platform
import subprocess

def check_docker_version():
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        print("Docker Version:", result.stdout.strip())
    except Exception as e:
        print("Docker not found:", e)

def main():
    print("=== Environment Verification Script ===")
    print("System:", platform.system(), platform.release())
    check_docker_version()

if __name__ == "__main__":
    main()

