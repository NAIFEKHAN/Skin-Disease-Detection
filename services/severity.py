"""
Prototype Severity Estimation Service.
Rule-based prototype risk categorization for educational and demonstration purposes.
"""

from typing import Dict, Any, Optional, List


class SeverityService:
    """
    Computes a rule-based prototype severity indicator.

    CRITICAL CLINICAL BOUNDARIES:
    - This is a heuristic prototype indicator for academic demonstration only.
    - It does NOT infer actual cancer stage, Breslow depth, or clinical prognosis.
    - Low model confidence or diagnostic ambiguity automatically triggers 'Undetermined'.
    - Benign predictions must never be interpreted as clinical confirmation that malignancy is absent.
    """

    # Disease risk tiers for algorithmic demonstration
    HIGH_RISK_CLASSES = {"mel"}          # Melanoma (highest malignancy potential)
    MODERATE_RISK_CLASSES = {"bcc", "akiec"}  # Malignant locally invasive or precancerous
    LOW_RISK_CLASSES = {"nv", "bkl", "df", "vasc"}  # Typically benign entities

    def estimate_severity(
        self,
        predicted_class: str,
        confidence_score: float,
        top_3_predictions: Optional[List[Dict[str, Any]]] = None,
        relative_lesion_area: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Determines prototype severity based on predicted disease entity, classification
        confidence, and presence of malignant differential diagnoses.
        """
        c_code = predicted_class.lower()

        # Check if Melanoma is present in top differentials
        mel_in_diff = False
        mel_prob = 0.0
        if top_3_predictions:
            for pred in top_3_predictions:
                if pred.get("class_code", "").lower() == "mel":
                    mel_in_diff = True
                    mel_prob = pred.get("probability", 0.0)

        # 1. Ambiguity / Low Confidence check
        # If model is uncertain (< 50% confidence) or melanoma is an active differential (> 15%)
        if confidence_score < 0.50 or (c_code != "mel" and mel_in_diff and mel_prob > 0.15):
            level = "Undetermined"
            badge_color = "#e67e22"  # Amber/Orange
            summary = "Undetermined — Clinical Examination Required"
            reason = (
                "The algorithmic prediction exhibits low confidence or presents elevated diagnostic ambiguity "
                "with high-risk categories in the differential. A qualified dermatologist must perform direct dermoscopy."
            )
            recommendation = "Schedule an in-person dermatological evaluation with dermoscopic examination."

        # 2. High Risk entities (Melanoma)
        elif c_code in self.HIGH_RISK_CLASSES:
            level = "High Concern"
            badge_color = "#c0392b"  # Red
            summary = "High Priority — Potential Malignancy"
            reason = (
                "The model predicted features consistent with Melanoma, an aggressive skin cancer requiring urgent "
                "histopathological verification (excisional biopsy)."
            )
            recommendation = "Urgent consultation with an onco-dermatologist for confirmatory biopsy and margin staging."

        # 3. Moderate Risk entities (BCC, AKIEC)
        elif c_code in self.MODERATE_RISK_CLASSES:
            level = "Moderate Concern"
            badge_color = "#d35400"  # Dark Amber
            summary = "Moderate Priority — Precancerous / Locally Invasive"
            if c_code == "bcc":
                reason = "Predicted Basal Cell Carcinoma; slowly progressive, locally destructive, requiring excision."
                recommendation = "Consult a dermatologist for surgical assessment, curettage, or cryosurgical intervention."
            else:
                reason = "Predicted Actinic Keratosis; precancerous epidermal lesion with potential progression to squamous cell carcinoma."
                recommendation = "Consult a dermatologist for topical field therapy or cryotherapy."

        # 4. Low Risk / Benign entities (NV, BKL, DF, VASC)
        else:
            level = "Low Concern"
            badge_color = "#27ae60"  # Green
            summary = "Low Priority — Characteristically Benign Pattern"
            reason = (
                "Features align with common benign dermatological entities. However, clinical changes (ABCDE criteria) "
                "warrant professional re-evaluation."
            )
            recommendation = "Routine observation. If you observe changes in border, size, symmetry, or color, consult a physician."

        disclaimer = (
            "Prototype algorithmic risk rating for educational demonstration only. Not a medical assessment, "
            "clinical staging, or confirmation that malignancy is absent. Always consult a certified healthcare professional."
        )

        return {
            "level": level,
            "badge_color": badge_color,
            "summary": summary,
            "reason": reason,
            "recommendation": recommendation,
            "is_prototype": True,
            "disclaimer": disclaimer
        }


# Singleton instance
_severity_service_instance: Optional[SeverityService] = None


def get_severity_service() -> SeverityService:
    """Returns singleton instance of SeverityService."""
    global _severity_service_instance
    if _severity_service_instance is None:
        _severity_service_instance = SeverityService()
    return _severity_service_instance
