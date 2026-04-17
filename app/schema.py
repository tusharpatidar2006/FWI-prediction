from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    """
    Input schema for the Fire Weather Index (FWI) prediction endpoint.
    Features must be provided in this exact order internally.
    """

    Temperature: float = Field(..., description="Temperature in Celsius", example=29.0)
    RH: float = Field(..., description="Relative Humidity in %", example=57.0)
    Ws: float = Field(..., description="Wind speed in km/h", example=18.0)
    Rain: float = Field(..., description="Rainfall in mm", example=0.0)
    FFMC: float = Field(..., description="Fine Fuel Moisture Code", example=65.7)
    DMC: float = Field(..., description="Duff Moisture Code", example=3.4)
    DC: float = Field(..., description="Drought Code", example=7.6)
    ISI: float = Field(..., description="Initial Spread Index", example=1.3)
    BUI: float = Field(..., description="Build Up Index", example=3.4)

    model_config = {
        "json_schema_extra": {
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
    }


class PredictionOutput(BaseModel):
    """
    Output schema returned after a successful prediction.
    """

    prediction: float = Field(..., description="Predicted FWI score")
    status: str = Field(default="success", description="Request status")


class HealthResponse(BaseModel):
    """
    Response schema for the /health endpoint.
    """

    status: str
    model_loaded: bool
    scaler_loaded: bool
    message: str