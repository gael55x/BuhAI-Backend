import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Date, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class ChatTurn(Base):
    __tablename__ = 'chat_turns'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=func.now())
    actor = Column(String, nullable=False) # 'user' or 'assistant'
    message = Column(Text, nullable=False)

class CGMStream(Base):
    __tablename__ = 'cgm_stream'
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, unique=True, nullable=False)
    glucose_level = Column(Float, nullable=True)
    sensor_id = Column(String)
    signal_quality_flag = Column(String)
    glucose_trend = Column(String)

class MealEvent(Base):
    __tablename__ = 'meal_events'
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, unique=True, nullable=False)
    date = Column(Date)
    meal_type = Column(String)
    food_items = Column(Text)  # Storing list as a string
    portion_estimates = Column(Text) # Storing list as a string
    baseline_glucose = Column(Float)
    sleep_quality = Column(String)
    glucose_at_t_30min = Column('glucose_at_t+30min', Float)
    glucose_at_t_60min = Column('glucose_at_t+60min', Float)
    auc_postprandial_2h = Column('AUC_postprandial_2h', Float)
    return_to_baseline_time = Column(Float)
    next_hypo_risk = Column(Float)
    next_hyper_risk = Column(Float)

class ActivityLog(Base):
    __tablename__ = 'activity_logs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp_start = Column(DateTime, unique=True, nullable=False)
    activity_type = Column(String)
    duration_min = Column(Integer)
    intensity = Column(String)
    steps = Column(Integer)

class SleepLog(Base):
    __tablename__ = 'sleep_logs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    sleep_start = Column(DateTime, unique=True, nullable=False)
    date = Column(Date)
    sleep_end = Column(DateTime)
    duration_h = Column(Float)
    num_wakeups = Column(Integer)
    sleep_quality = Column(String)
    was_disrupted = Column(Boolean)

class CGMAggregate(Base):
    __tablename__ = 'cgm_aggregates'
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, unique=True, nullable=False)
    mean_glucose = Column(Float)
    gmi = Column(Float)
    time_in_range_pct = Column(Float)
    cv = Column(Float)
    sd = Column(Float)
    mage = Column(Float)
    hypo_flag = Column(Integer)
    hyper_flag = Column(Integer)

    def to_dict(self):
        """Converts the object to a dictionary, making date JSON serializable."""
        return {
            "id": self.id,
            "date": self.date.isoformat() if self.date else None,
            "mean_glucose": self.mean_glucose,
            "gmi": self.gmi,
            "time_in_range_pct": self.time_in_range_pct,
            "cv": self.cv,
            "sd": self.sd,
            "mage": self.mage,
            "hypo_flag": self.hypo_flag,
            "hyper_flag": self.hyper_flag,
        }

class MedicalIntake(Base):
    __tablename__ = 'medical_intake'
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now())
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    
def init_db(db_path="data/buhai.db"):
    """Initializes the database, creating it and the tables if they don't exist."""
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    
    logger.info(f"Initializing database at {db_path}...")
    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)
    logger.info("Database and tables created successfully.")
    
    Session = sessionmaker(bind=engine)
    return Session

if __name__ == '__main__':
    # Example of how to use it
    DB_SESSION = init_db()
    session = DB_SESSION()
    # You can add some initial data here if needed
    session.close() 