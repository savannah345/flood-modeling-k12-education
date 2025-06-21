import bcrypt
import psycopg
import streamlit as st
import supabase
from supabase import create_client

# === DATABASE AUTH ===
SUPABASE_DB_URL = st.secrets["SUPABASE_DB_URL"]
conn = psycopg.connect(SUPABASE_DB_URL)
cur = conn.cursor()

# === AUTH HELPERS ===
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def create_user(email, password):
    hashed = hash_password(password)
    try:
        cur.execute("INSERT INTO users (email, password) VALUES (%s, %s)", (email, hashed))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print("Create user error:", e)
        return False

def authenticate_user(email, password):
    cur.execute("SELECT id, password FROM users WHERE email = %s", (email,))
    result = cur.fetchone()
    if result and check_password(password, result[1]):
        return result[0]
    return None

def reset_password(email, new_password):
    hashed = hash_password(new_password)
    try:
        cur.execute("UPDATE users SET password = %s WHERE email = %s", (hashed, email))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print("Reset error:", e)
        return False


def delete_user_files(user_id: str):
    try:
        files = supabase.storage.from_("swmm-files").list(path=user_id)
        for f in files:
            supabase.storage.from_("swmm-files").remove([f"{user_id}/{f['name']}"])
        return True
    except Exception as e:
        print(f"File cleanup error for {user_id}:", e)
        return False
