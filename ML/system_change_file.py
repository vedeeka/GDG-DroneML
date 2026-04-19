import os

ENV_FILE = ".env"

def read_env():
    env_data = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    env_data[key] = value
    return env_data

def write_env(env_data):
    with open(ENV_FILE, "w") as f:
        for key, value in env_data.items():
            f.write(f"{key}={value}\n")

def update_env():
    env_data = read_env()

    print("Enter values (leave blank to keep existing):\n")

    # Define fields you want
    fields = ["email"]  # Add more fields as needed

    for field in fields:
        current_value = env_data.get(field, "")
        user_input = input(f"{field} [{current_value}]: ")

        if user_input.strip() != "":
            env_data[field] = user_input  # update
        else:
            env_data[field] = current_value  # keep old

    write_env(env_data)
    print("\n✅ .env file updated successfully!")

if __name__ == "__main__":
    update_env()