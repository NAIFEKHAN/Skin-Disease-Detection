"""
Treatment Cost Estimation Service (in Indian Rupees - INR).
Provides illustrative cost breakdowns for college mini-project demonstration.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

DEFAULT_COST_FILE = Path(__file__).resolve().parent.parent / "data" / "treatment_costs.json"


class CostEstimationService:
    """
    Computes illustrative dermatological cost estimations in INR.

    IMPORTANT DISCLAIMER:
    - Costs are strictly illustrative educational estimates for typical Indian clinical settings.
    - They are NOT actual hospital quotations, medical bills, or insurance estimates.
    - Specific clinical treatment pathways and medications must be determined exclusively by a licensed doctor.
    """

    def __init__(self, cost_file_path: Optional[Path] = None):
        self.cost_file_path = cost_file_path if cost_file_path else DEFAULT_COST_FILE
        self.data: Dict[str, Any] = {}
        self._load_data()

    def _load_data(self):
        """Loads cost benchmarks from JSON configuration."""
        if not self.cost_file_path.exists():
            raise FileNotFoundError(f"Treatment costs configuration not found at: {self.cost_file_path}")

        with open(self.cost_file_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def estimate_cost(self, predicted_class: str) -> Dict[str, Any]:
        """
        Retrieves illustrative cost breakdown for the given disease class code.
        """
        c_code = predicted_class.lower()
        categories = self.data.get("categories", {})

        if c_code not in categories:
            # Fallback to general consultation
            c_data = categories.get("nv", {})
        else:
            c_data = categories[c_code]

        currency = self.data.get("currency", "INR")
        symbol = self.data.get("currency_symbol", "₹")

        breakdown = [
            {
                "category": "Doctor Consultation",
                "description": c_data["consultation"]["description"],
                "min_cost": c_data["consultation"]["min_cost"],
                "max_cost": c_data["consultation"]["max_cost"],
                "formatted_range": f"{symbol}{c_data['consultation']['min_cost']:,} – {symbol}{c_data['consultation']['max_cost']:,}"
            },
            {
                "category": "Diagnostic Tests",
                "description": c_data["diagnostics"]["description"],
                "min_cost": c_data["diagnostics"]["min_cost"],
                "max_cost": c_data["diagnostics"]["max_cost"],
                "formatted_range": f"{symbol}{c_data['diagnostics']['min_cost']:,} – {symbol}{c_data['diagnostics']['max_cost']:,}"
            },
            {
                "category": "Medication & Dressings",
                "description": c_data["medication"]["description"],
                "min_cost": c_data["medication"]["min_cost"],
                "max_cost": c_data["medication"]["max_cost"],
                "formatted_range": f"{symbol}{c_data['medication']['min_cost']:,} – {symbol}{c_data['medication']['max_cost']:,}"
            },
            {
                "category": "Procedure / Intervention",
                "description": c_data["procedure"]["description"],
                "min_cost": c_data["procedure"]["min_cost"],
                "max_cost": c_data["procedure"]["max_cost"],
                "formatted_range": f"{symbol}{c_data['procedure']['min_cost']:,} – {symbol}{c_data['procedure']['max_cost']:,}"
            },
            {
                "category": "Follow-Up Consultation",
                "description": c_data["follow_up"]["description"],
                "min_cost": c_data["follow_up"]["min_cost"],
                "max_cost": c_data["follow_up"]["max_cost"],
                "formatted_range": f"{symbol}{c_data['follow_up']['min_cost']:,} – {symbol}{c_data['follow_up']['max_cost']:,}"
            }
        ]

        total_min = c_data.get("total_min", sum(item["min_cost"] for item in breakdown))
        total_max = c_data.get("total_max", sum(item["max_cost"] for item in breakdown))

        return {
            "currency": currency,
            "currency_symbol": symbol,
            "disease_code": c_code.upper(),
            "disease_name": c_data.get("disease_name", c_code.upper()),
            "nature": c_data.get("nature", "Dermatological Lesion"),
            "total_min": total_min,
            "total_max": total_max,
            "formatted_total_range": f"{symbol}{total_min:,} – {symbol}{total_max:,}",
            "breakdown": breakdown,
            "disclaimer": self.data.get("disclaimer", "")
        }


# Singleton instance
_cost_service_instance: Optional[CostEstimationService] = None


def get_cost_service() -> CostEstimationService:
    """Returns singleton instance of CostEstimationService."""
    global _cost_service_instance
    if _cost_service_instance is None:
        _cost_service_instance = CostEstimationService()
    return _cost_service_instance
