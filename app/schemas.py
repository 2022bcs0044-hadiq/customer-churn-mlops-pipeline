from typing import Union
from pydantic import BaseModel, Field, field_validator


class CustomerData(BaseModel):
    """
    Schema for customer data payload supporting both full feature specifications
    and minimal payloads with backward-compatible defaults and type coercions.
    """
    # Demographics
    gender: Union[str, int] = Field(
        default="Female",
        description="Customer gender: 'Female', 'Male', or 0 (Female) / 1 (Male)"
    )
    SeniorCitizen: int = Field(
        default=0, ge=0, le=1,
        description="1 if customer is a senior citizen, else 0"
    )
    Partner: Union[str, int] = Field(
        default=0,
        description="1 or 'Yes' if customer has a partner, else 0 or 'No'"
    )
    Dependents: Union[str, int] = Field(
        default=0,
        description="1 or 'Yes' if customer has dependents, else 0 or 'No'"
    )

    # Account tenure & Phone services
    tenure: int = Field(
        default=1, ge=0,
        description="Number of months customer has stayed with company"
    )
    PhoneService: Union[str, int] = Field(
        default=1,
        description="1 or 'Yes' if customer has phone service, else 0 or 'No'"
    )
    MultipleLines: str = Field(
        default="No",
        description="'No', 'Yes', or 'No phone service'"
    )

    # Internet & Value-added services
    InternetService: str = Field(
        default="Fiber optic",
        description="'DSL', 'Fiber optic', or 'No'"
    )
    OnlineSecurity: str = Field(
        default="No",
        description="'Yes', 'No', or 'No internet service'"
    )
    OnlineBackup: str = Field(
        default="No",
        description="'Yes', 'No', or 'No internet service'"
    )
    DeviceProtection: str = Field(
        default="No",
        description="'Yes', 'No', or 'No internet service'"
    )
    TechSupport: str = Field(
        default="No",
        description="'Yes', 'No', or 'No internet service'"
    )
    StreamingTV: str = Field(
        default="No",
        description="'Yes', 'No', or 'No internet service'"
    )
    StreamingMovies: str = Field(
        default="No",
        description="'Yes', 'No', or 'No internet service'"
    )

    # Contract, Billing & Charges
    Contract: str = Field(
        default="Month-to-month",
        description="'Month-to-month', 'One year', or 'Two year'"
    )
    PaperlessBilling: Union[str, int] = Field(
        default=1,
        description="1 or 'Yes' if customer uses paperless billing, else 0 or 'No'"
    )
    PaymentMethod: str = Field(
        default="Electronic check",
        description="'Electronic check', 'Mailed check', 'Bank transfer (automatic)', or 'Credit card (automatic)'"
    )
    MonthlyCharges: float = Field(
        default=50.0, ge=0.0,
        description="The amount charged to the customer monthly"
    )
    TotalCharges: float = Field(
        default=50.0, ge=0.0,
        description="The total amount charged to the customer"
    )

    @field_validator("gender", mode="before")
    @classmethod
    def normalize_gender(cls, v):
        if v == 1 or v == "1":
            return "Male"
        if v == 0 or v == "0":
            return "Female"
        if isinstance(v, str):
            return v.capitalize()
        return v

    @field_validator("Partner", "Dependents", "PhoneService", "PaperlessBilling", mode="before")
    @classmethod
    def normalize_binary(cls, v):
        if isinstance(v, str):
            return 1 if v.strip().lower() in ("yes", "true", "1") else 0
        return int(v)