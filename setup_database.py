#!/usr/bin/env python3
"""
Database setup script for IT Support Chatbot
This script creates the necessary tables in your NeonDB PostgreSQL instance
"""
import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from database import Base

def setup_database():
    """Setup database tables"""
    # Load environment variables
    load_dotenv()
    
    # Get database URL from environment
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ Error: DATABASE_URL not found in environment variables")
        print("Please set DATABASE_URL in your .env file")
        print("Example: postgresql://username:password@host:port/database")
        sys.exit(1)
    
    try:
        # Create engine
        print("🔗 Connecting to database...")
        engine = create_engine(database_url)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ Database connection successful!")
        
        # Create all tables
        print("📋 Creating database tables...")
        Base.metadata.create_all(bind=engine)
        
        # Verify tables were created
        with engine.connect() as conn:
            # Check if users table exists
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'users'
                );
            """))
            users_exists = result.scalar()
            
            # Check if tickets table exists
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'tickets'
                );
            """))
            tickets_exists = result.scalar()
            
            if users_exists and tickets_exists:
                print("✅ All tables created successfully!")
                print("📊 Tables created:")
                print("   - users (user_id, name, created_at, updated_at)")
                print("   - tickets (ticket_id, user_id, problem_description, solution_suggested, status, created_at, updated_at)")
            else:
                print("❌ Error: Some tables were not created")
                sys.exit(1)
        
        print("\n🎉 Database setup completed successfully!")
        print("You can now run the FastAPI application with: python run.py")
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your DATABASE_URL format")
        print("2. Ensure your NeonDB instance is running")
        print("3. Verify your database credentials")
        print("4. Check if your IP is whitelisted in NeonDB")
        sys.exit(1)

def show_table_structure():
    """Show the structure of created tables"""
    print("\n📋 Database Schema:")
    print("=" * 50)
    
    print("\n🔹 Users Table:")
    print("   user_id (String, Primary Key) - User's unique identifier")
    print("   name (String) - User's display name")
    print("   created_at (DateTime) - When user was created")
    print("   updated_at (DateTime) - When user was last updated")
    
    print("\n🔹 Tickets Table:")
    print("   ticket_id (Integer, Primary Key, Auto-increment) - Unique ticket ID")
    print("   user_id (String, Foreign Key) - References users.user_id")
    print("   problem_description (Text) - Description of the IT problem")
    print("   solution_suggested (Text) - AI-suggested solution")
    print("   status (String, Default: 'open') - Ticket status")
    print("   created_at (DateTime) - When ticket was created")
    print("   updated_at (DateTime) - When ticket was last updated")

if __name__ == "__main__":
    print("🚀 IT Support Chatbot - Database Setup")
    print("=" * 50)
    
    setup_database()
    show_table_structure()
