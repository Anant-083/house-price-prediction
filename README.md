# 🏠 House Price Prediction

A Machine Learning web application that predicts house prices based on various features using Linear Regression.

## 🔗 Live Demo
https://house-price-prediction-4jezb9hh9hbiw9bykkwnfy.streamlit.app/

## 📊 About The Project
This project uses the California Housing Dataset containing 20,640 real house records to train a Linear Regression model that predicts house prices.

## 🎯 Model Performance
- **Algorithm:** Linear Regression
- **Accuracy:** 57.6%
- **R2 Score:** 0.58
- **Dataset Size:** 20,640 records
- **Training Samples:** 16,512
- **Testing Samples:** 4,128

## 🛠️ Technologies Used
- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

## 📁 Project Structure
```
house-price-prediction/
│
├── house_price.py      ← ML model code
├── app.py              ← Streamlit web app
├── results.png         ← Model results graph
└── README.md           ← Project documentation
```

## 🚀 How To Run Locally
```bash
# Clone the repository
git clone https://github.com/Anant-083/house-price-prediction.git

# Navigate to folder
cd house-price-prediction

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn streamlit

# Run the app
streamlit run app.py
```

## 📈 Results
![Actual vs Predicted Prices](results.png)

## 👨‍💻 Author
**Anant-083**
- GitHub: [@Anant-083](https://github.com/Anant-083)

## 📚 What I Learned
- End to end ML pipeline
- Data preprocessing and exploration
- Training and evaluating ML models
- Building and deploying ML web apps
