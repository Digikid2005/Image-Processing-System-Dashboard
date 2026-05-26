from app import app
from models import db, WebUser

def reset_database():
    with app.app_context():
        print("[SYSTEM] Dropping legacy database tables...")
        db.drop_all()
        
        print("[SYSTEM] Building new SQLAlchemy schema...")
        db.create_all()
        
        print("[SYSTEM] Re-provisioning default admin account...")
        admin_user = WebUser(username='admin', password_hash='admin123')
        db.session.add(admin_user)
        db.session.commit()
        
        print("[SYSTEM] Database migration completed successfully.")

if __name__ == "__main__":
    reset_database()
