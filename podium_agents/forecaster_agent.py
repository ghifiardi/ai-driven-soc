import random

class ForecasterAgent:
    def __init__(self):
        self.latest_predictions = {} # For dashboard
        self.forecasting_models = {
            "short_term": "prophet",
            "medium_term": "arima",
            "long_term": "neural_prophet"
        }
    
    def generate_forecasts(self, cleaned_data):
        """Menghasilkan forecast menggunakan data tanpa anomali"""
        forecasts = {}
        
        for horizon, model in self.forecasting_models.items():
            # Persiapan data - menghilangkan anomali
            clean_series = self.remove_anomalies_from_series(cleaned_data)
            
            # Generate forecast
            forecast = self.apply_forecast_model(
                model=model,
                data=clean_series,
                horizon=self.get_horizon(horizon)
            )
            
            # Tambahkan confidence intervals menggunakan Cortex
            forecast_with_ci = self.add_confidence_intervals(forecast)
            
            forecasts[horizon] = forecast_with_ci
        
        # Gabungkan forecast untuk pandangan komprehensif
        consolidated = self.consolidate_forecasts(forecasts)
        self.latest_predictions = consolidated
        return consolidated
    
    def remove_anomalies_from_series(self, data_series):
        """Membersihkan data dari anomali untuk forecasting yang akurat"""
        # Identifikasi dan hapus outlier
        cleaned_data = self.apply_anomaly_filter(
            data=data_series,
            method="cortex_anomaly_detection",
            threshold=0.8
        )
        
        # Imputasi nilai yang hilang (jika diperlukan)
        imputed_data = self.impute_missing_values(cleaned_data)
        
        return imputed_data

    # --- MOCKED METHODS ---

    def get_horizon(self, horizon_name):
        return {"short_term": 7, "medium_term": 30, "long_term": 90}.get(horizon_name, 7)

    def apply_anomaly_filter(self, data, method, threshold):
        # Mock filter
        return data

    def impute_missing_values(self, data):
        return data

    def apply_forecast_model(self, model, data, horizon):
        # Mock forecast data
        return [random.randint(100, 200) for _ in range(horizon)]

    def add_confidence_intervals(self, forecast):
        return {"values": forecast, "lower": 90, "upper": 210}

    def consolidate_forecasts(self, forecasts):
        return {"summary": "Stable growth", "details": forecasts}
