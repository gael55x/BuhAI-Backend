# BuhAI-Backend

A comprehensive diabetes management backend with AI-powered chat capabilities and LSTM glucose prediction models.

## Features

### AI Chat Assistant
- **RAG (Retrieval-Augmented Generation)** system for personalized diabetes advice
- **ChromaDB Vector Store** for fast semantic retrieval of user data
- **Gemini AI Integration** for natural language processing in Bisaya/English
- Real-time chat API for continuous diabetes support

### LSTM Glucose Prediction
- **Multivariate LSTM Models** for 30-minute and 60-minute glucose predictions
- **Feature Engineering** including meal flags, activity levels, sleep quality, and time patterns
- **Pre-trained Models** ready for immediate use
- **RESTful API** endpoints for easy integration

### Data Management
- **SQLite Database** for structured health data storage
- **CSV Data Ingestion** for bulk data imports
- **User Activity Logs** tracking meals, exercise, sleep, and glucose readings

## Architecture

```
BuhAI-Backend/
├── api/                    # REST API endpoints
│   ├── chat_blueprint.py   # AI chat endpoints
│   └── prediction_blueprint.py  # LSTM prediction endpoints
├── model/                  # LSTM prediction models
│   ├── lstm_models.py      # Training pipeline
│   ├── prediction_utils.py # Prediction utilities
│   ├── models/             # Pre-trained model files
│   │   ├── lstm_mv_30.h5   # 30-minute prediction model
│   │   ├── lstm_mv_60.h5   # 60-minute prediction model
│   │   └── scaler_mv.pkl   # Feature scaler
│   └── run.sh              # Training script
├── rag/                    # RAG system components
│   ├── chat_logic.py       # Chat processing logic
│   ├── data_handler.py     # Data retrieval and processing
│   ├── llm.py              # Language model interface
│   └── retriever.py        # Vector store retrieval
├── data/                   # Data storage
│   ├── buhai.db           # SQLite database
│   └── dataset-user/      # CSV data files
└── vector_store/          # ChromaDB vector embeddings
```

## Quick Start

### Prerequisites
- Python 3.9+
- TensorFlow 2.x
- Flask
- ChromaDB
- Gemini API Key

### Installation

1. **Clone and navigate to directory**
   ```bash
   cd BuhAI-Backend
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

The server will start on `http://localhost:4000`

## API Endpoints

### Chat API
- `POST /api/v1/chat` - Send a message to the AI assistant
- `GET /api/v1/chat/health` - Check chat service health

### LSTM Prediction API
- `POST /api/v1/predict` - Make glucose predictions
- `GET /api/v1/predict/health` - Check prediction service health
- `GET /api/v1/predict/info` - Get model information
- `GET /api/v1/predict/sample` - Test with sample data

### Example Prediction Request
```bash
curl -X POST http://localhost:4000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "glucose_readings": [120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205],
    "meal_flags": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "activity_levels": [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "sleep_quality": 0,
    "horizon": "both"
  }'
```

### Example Response
```json
{
  "success": true,
  "predictions": {
    "30min": 185.2,
    "60min": 192.8
  },
  "message": "Predictions generated successfully"
}
```

## LSTM Model Details

### Model Architecture
- **2-layer LSTM** with 64 units each for primary model
- **1-layer GRU** with 64 units for comparison (automatically selects best)
- **Dense output layer** for glucose prediction

### Features Used
1. **glucose_level** - Current glucose readings
2. **meal_flag_hiGI** - High glycemic index meal indicator
3. **activity_intensity** - Exercise intensity (0=low, 1=medium, 2=high)
4. **sleep_quality** - Sleep quality (0=good, 1=poor)
5. **hour_sin/hour_cos** - Cyclical time features

### Model Performance
- **30-minute predictions**: Optimized for short-term glucose forecasting
- **60-minute predictions**: Extended horizon for meal and activity planning
- **Input requirements**: 18 glucose readings (90 minutes of 5-minute interval data)

## Training New Models

To retrain the LSTM models with your own data:

1. **Prepare your data** in the `data/dataset-user/` directory:
   - `cgm_stream.csv` - Glucose readings with timestamps
   - `meal_events.csv` - Meal logs with glycemic impact
   - `activity_logs.csv` - Exercise logs with intensity
   - `sleep_logs.csv` - Sleep quality data

2. **Run the training script**:
   ```bash
   cd model/
   chmod +x run.sh
   ./run.sh
   ```

3. **Monitor training progress** in `model/model.logs`

## Testing

Run the integration test to verify everything is working:

```bash
python test_lstm_integration.py
```

This will test:
- ✅ Model loading
- ✅ Prediction generation  
- ✅ API endpoint functionality

## Data Requirements

### Glucose Data Format
```csv
timestamp,glucose_level
2024-01-01 00:00:00,120
2024-01-01 00:05:00,125
```

### Meal Data Format
```csv
timestamp,next_hyper_risk
2024-01-01 08:00:00,1
2024-01-01 12:00:00,0
```

### Activity Data Format
```csv
timestamp_start,duration_min,intensity
2024-01-01 09:00:00,30,medium
2024-01-01 18:00:00,45,high
```

### Sleep Data Format
```csv
sleep_start,sleep_end,sleep_quality
2024-01-01 22:00:00,2024-01-02 06:00:00,good
2024-01-02 23:00:00,2024-01-03 07:00:00,poor
```

## Development

### Adding New Features
1. **Chat Features**: Extend `rag/chat_logic.py`
2. **Prediction Features**: Modify `model/prediction_utils.py`
3. **API Endpoints**: Add new blueprints in `api/`

### Database Schema
The SQLite database stores:
- User profiles and preferences
- Historical glucose readings
- Meal, activity, and sleep logs
- Chat conversation history

## Deployment

For production deployment:

1. **Set environment variables**:
   ```bash
   export FLASK_ENV=production
   export DATABASE_URL=your_production_db_url
   export GEMINI_API_KEY=your_api_key
   ```

2. **Use a production WSGI server**:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:4000 app:app
   ```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request

## License

This project is part of the BuhAI diabetes management system.

---

**Note**: This backend is designed to work with the BuhAI React Native frontend. For the complete diabetes management solution, deploy both components together.