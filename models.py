from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Initialize the SQLAlchemy instance
db = SQLAlchemy()

class WebUser(db.Model):
    __tablename__ = 'web_users'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<WebUser(username='{self.username}')>"

class CapturedImage(db.Model):
    __tablename__ = 'captured_images'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    is_processed = db.Column(db.Boolean, default=False, nullable=False)
    is_analyzed = db.Column(db.Boolean, default=False, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<CapturedImage(filename='{self.filename}', is_processed={self.is_processed})>"

