from pydantic import BaseModel, Field


class FireFeaturesInput(BaseModel):
    """
    Input features for Algerian Forest Fire FWI (Fire Weather Index) prediction.

    9 weather/fuel-moisture features after correlation-based feature removal
    (threshold=0.85). The 'Classes' column is excluded — the scaler was
    fitted on these 9 numerical weather features only.
    """

    Temperature: float = Field(..., example=29.0, description="Max temperature in Celsius")
    RH: float = Field(..., example=57.0, description="Relative Humidity in %")
    Ws: float = Field(..., example=18.0, description="Wind speed in km/h")
    Rain: float = Field(..., example=0.0, description="Total rainfall in mm")
    FFMC: float = Field(..., example=65.7, description="Fine Fuel Moisture Code index")
    DMC: float = Field(..., example=3.4, description="Duff Moisture Code index")
    DC: float = Field(..., example=7.6, description="Drought Code index")
    ISI: float = Field(..., example=1.3, description="Initial Spread Index")
    BUI: float = Field(..., example=3.4, description="Buildup Index")

    class Config:
        json_schema_extra = {
            "example": {
                "Temperature": 29.0,
                "RH": 57.0,
                "Ws": 18.0,
                "Rain": 0.0,
                "FFMC": 65.7,
                "DMC": 3.4,
                "DC": 7.6,
                "ISI": 1.3,
                "BUI": 3.4,
            }
        }


class PredictionResponse(BaseModel):
    fwi_prediction: float = Field(..., description="Predicted Fire Weather Index (FWI)")
    status: str = Field(..., description="Prediction status message")