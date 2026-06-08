# NASA POWER Wildfire Risk Assessment - Deployment Guide

## Quick Start (Local Development)

### 1. Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (for NASA POWER API)

### 2. Installation Steps

1. **Create project directory:**
   ```bash
   mkdir wildfire-risk-app
   cd wildfire-risk-app
   ```

2. **Save the files:**
   - Save `requirements.txt` in the project directory
   - Save `wildfire_risk_app.py` in the project directory
   - Save `setup.py` in the project directory

3. **Run setup:**
   ```bash
   python setup.py
   ```

4. **Start the application:**
   ```bash
   streamlit run wildfire_risk_app.py
   ```

5. **Access the app:**
   - Open your web browser to `http://localhost:8501`
   - The app should load automatically

## Features of Your Web App

### Current Features
- **Interactive Map**: Folium-based map with drawing tools
- **Data Visualization**: Heatmaps, time series, and animations
- **Export Functionality**: CSV download of filtered data
- **Parameter Selection**: Multiple meteorological parameters
- **Date Range Filtering**: Flexible date selection
- **Sample Data**: Demo mode with realistic synthetic data

### Features to Implement
- **NASA POWER API Integration**: Live data fetching
- **Authentication**: User accounts (if needed)
- **Advanced Analytics**: Statistical analysis tools
- **Multi-layer Visualization**: Multiple parameters simultaneously

## Deployment Options

### Option 1: Local Development Server (Easiest)
**Best for**: Testing, development, personal use
**Cost**: Free
**Steps**: Follow the Quick Start guide above

### Option 2: Streamlit Cloud (Recommended)
**Best for**: Sharing with others, public deployment
**Cost**: Free tier available
**Steps**:
1. Push your code to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy your repository

### Option 3: Heroku
**Best for**: Professional deployment with custom domain
**Cost**: $7/month minimum
**Steps**:
1. Create `Procfile`: `web: streamlit run wildfire_risk_app.py --server.port=$PORT --server.address=0.0.0.0`
2. Create Heroku app: `heroku create your-app-name`
3. Deploy: `git push heroku main`

### Option 4: AWS/Azure/GCP
**Best for**: Enterprise deployment, high traffic
**Cost**: Variable (pay-as-you-use)
**Steps**: Requires containerization with Docker

## File Structure
```
wildfire-risk-app/
├── wildfire_risk_app.py      # Main application
├── requirements.txt          # Python dependencies
├── setup.py                 # Setup script
├── DEPLOYMENT_GUIDE.md      # This file
└── README.md               # Project documentation
```

## Configuration Options

### Environment Variables (Optional)
Create a `.env` file for configuration:
```env
# NASA POWER API settings
NASA_POWER_TIMEOUT=30
DEFAULT_GRID_SIZE=20

# Map settings
DEFAULT_LAT=41.25
DEFAULT_LON=-77.5
DEFAULT_ZOOM=7

# Data settings
MAX_DATE_RANGE_DAYS=365
```

### Customization Points
1. **Map Center**: Change default coordinates in the code
2. **Available Parameters**: Modify `AVAILABLE_PARAMETERS` dictionary
3. **Color Schemes**: Update `make_color_scale()` function
4. **Grid Resolution**: Adjust `grid_size` parameter

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   streamlit run wildfire_risk_app.py --server.port 8502
   ```

2. **Module not found errors**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **Map not loading**
   - Check internet connection
   - Try refreshing the page
   - Clear browser cache

4. **NASA API errors**
   - Check date ranges (API has limitations)
   - Verify coordinates are within valid ranges
   - Check API status at NASA POWER website

### Performance Optimization

1. **Large datasets**: Implement data pagination
2. **Slow loading**: Add caching with `@st.cache_data`
3. **Memory usage**: Limit grid resolution for large areas
4. **API rate limits**: Implement request throttling

## Next Steps for Enhancement

### Priority 1: Core Functionality
- [ ] Implement full NASA POWER API integration
- [ ] Add error handling for API failures  
- [ ] Optimize grid-based data fetching

### Priority 2: User Experience
- [ ] Add loading indicators
- [ ] Implement data validation
- [ ] Add help tooltips and tutorials

### Priority 3: Advanced Features
- [ ] User authentication and data persistence
- [ ] Advanced fire risk calculations
- [ ] Integration with other weather APIs
- [ ] Machine learning risk predictions

### Priority 4: Production Ready
- [ ] Comprehensive error handling
- [ ] Logging and monitoring
- [ ] Performance optimization
- [ ] Security hardening

## Support and Resources

- **Streamlit Documentation**: [docs.streamlit.io](https://docs.streamlit.io)
- **NASA POWER API**: [power.larc.nasa.gov](https://power.larc.nasa.gov)
- **Folium Documentation**: [python-visualization.github.io/folium](https://python-visualization.github.io/folium)
- **GeoPandas Documentation**: [geopandas.org](https://geopandas.org)

## Getting Help

If you encounter issues:
1. Check this deployment guide
2. Review error messages in the terminal
3. Check Streamlit logs for detailed errors
4. Test with sample data first before using live API

Good luck with your wildfire risk assessment application! 🔥📊
