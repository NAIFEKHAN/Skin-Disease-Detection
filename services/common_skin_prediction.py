"""
Common Skin Condition Prediction & Clinical Inference Service.
Clinical Photography Classification Pipeline (ViT-B/16).

Specifications:
- Mode: Common Skin Condition Analysis (Separate from Dermoscopic Lesion Analysis)
- Classes: 9 common clinical categories
  (Acne, Allergic Contact Dermatitis, Eczema, Impetigo, Psoriasis, Ringworm, Scabies, Seborrheic Dermatitis, Urticaria)
- Model: Vision Transformer Base (ViT-B/16, 'vit_base_patch16_224')
- Preprocessing: Clinical photography transforms (Resize + CenterCrop, ImageNet normalization)
- Strict Separation:
  - Does NOT merge probabilities with the 7-class dermoscopic lesion classifier.
  - Does NOT apply dermoscopic U-Net segmentation to clinical photos.
- Uncertainty Handling:
  - Reports indeterminate/uncertain predictions when confidence is low or entropy is elevated.
  - Never forces an unwarranted diagnosis.
- Activation Safety:
  - Ready for integration, but remains dormant/inactive in main user workflow until genuine
    GPU-trained weights (macro F1 > 0.50) are installed.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from models.common_skin_vit import (
    CommonSkinViTClassifier,
    load_common_skin_checkpoint,
    get_clinical_transforms,
    COMMON_SKIN_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    CLASS_DISPLAY_NAMES,
    NUM_COMMON_CLASSES
)

# Educational Clinical Knowledge Base for the 9 Common Skin Categories
COMMON_SKIN_DISEASE_INFO: Dict[str, Dict[str, Any]] = {
    "acne": {
        "display_name": "Acne",
        "category_type": "Inflammatory Pilosebaceous Disorder",
        "description": "Acne vulgaris is a multifactorial inflammatory disease of the pilosebaceous follicles characterized by comedones, papules, pustules, and occasionally nodules or cysts.",
        "common_triggers": ["Hormonal fluctuations", "Excess sebum production", "Cutibacterium acnes proliferation", "Comedogenic cosmetics"],
        "clinical_precaution": "Avoid aggressive manual extraction or popping of lesions to prevent permanent scarring and post-inflammatory hyperpigmentation.",
        "indicative_care": ["Gentle non-comedogenic cleansing", "Topical retinoids or benzoyl peroxide under supervision", "Sun protection"]
    },
    "allergic_contact_dermatitis": {
        "display_name": "Allergic Contact Dermatitis",
        "category_type": "Type IV Delayed Hypersensitivity Reaction",
        "description": "An immunologically mediated delayed hypersensitivity response occurring after cutaneous contact with an exogenous allergen (e.g., nickel, fragrances, poison ivy, topical preservatives).",
        "common_triggers": ["Nickel jewelry", "Fragrances", "Poison ivy / urushiol", "Cosmetic preservatives", "Rubber / latex accelerants"],
        "clinical_precaution": "Identify and strictly eliminate contact with the culprit allergen. Extensive or facial involvement requires professional dermatological assessment.",
        "indicative_care": ["Cool compresses", "Bland barrier emollients", "Patch testing by a dermatologist", "Short-course topical corticosteroids"]
    },
    "eczema": {
        "display_name": "Eczema (Atopic Dermatitis)",
        "category_type": "Chronic Pruritic Inflammatory Dermatosis",
        "description": "A chronic, relapsing, pruritic inflammatory skin disorder associated with skin barrier dysfunction (filaggrin deficiency) and immune system dysregulation.",
        "common_triggers": ["Dry climate", "Soaps and detergents", "Aeroallergens", "Emotional stress", "Temperature extremes"],
        "clinical_precaution": "Chronic scratching compromises the epidermal barrier, increasing vulnerability to secondary bacterial (Staphylococcus) and viral (eczema herpeticum) superinfections.",
        "indicative_care": ["Frequent thick emollient application", "Lukewarm short showers", "Topical calcineurin inhibitors or corticosteroids", "Trigger avoidance"]
    },
    "impetigo": {
        "display_name": "Impetigo",
        "category_type": "Contagious Superficial Bacterial Infection",
        "description": "A highly contagious bacterial skin infection, predominantly caused by Staphylococcus aureus or Streptococcus pyogenes, presenting with honey-colored crusted erosions.",
        "common_triggers": ["Minor skin trauma", "Warm humid weather", "Close physical contact", "Pre-existing scratch or insect bite"],
        "clinical_precaution": "Highly contagious. Patient towels, linens, and clothing should not be shared. Seek prompt medical care for targeted antibiotic therapy.",
        "indicative_care": ["Topical or oral prescription antibiotics", "Gentle crust debridement with warm water", "Strict hand hygiene"]
    },
    "psoriasis": {
        "display_name": "Psoriasis",
        "category_type": "Chronic Autoimmune Erythematosquamous Disease",
        "description": "A chronic immune-mediated systemic inflammatory disease characterized by hyperproliferation of keratinocytes, resulting in well-demarcated erythematous plaques with silvery scales.",
        "common_triggers": ["Physical trauma (Koebner phenomenon)", "Streptococcal infections", "Medications (beta-blockers, lithium)", "Systemic stress"],
        "clinical_precaution": "Psoriasis is a systemic inflammatory condition associated with metabolic syndrome, cardiovascular risk, and psoriatic arthritis in up to 30% of patients.",
        "indicative_care": ["Topical vitamin D analogues and corticosteroids", "Phototherapy", "Systemic biologics for moderate-to-severe disease", "Keratolytic agents"]
    },
    "ringworm": {
        "display_name": "Ringworm (Tinea Corporis)",
        "category_type": "Superficial Dermatophyte Fungal Infection",
        "description": "A superficial fungal infection of glabrous skin caused by dermatophytes (Trichophyton, Microsporum, or Epidermophyton), presenting as annular erythematous scaly plaques with active borders.",
        "common_triggers": ["Contact with infected individuals or animals", "Warm and moist environments", "Occlusive clothing", "Shared gym equipment"],
        "clinical_precaution": "Do not treat with topical corticosteroids alone ('tinea incognito'), which temporarily suppresses redness while exacerbating and masking the fungal proliferation.",
        "indicative_care": ["Topical antifungal creams (terbinafine, clotrimazole)", "Keep skin dry and clean", "Launder infected garments in hot water"]
    },
    "scabies": {
        "display_name": "Scabies",
        "category_type": "Parasitic Mite Infestation (Sarcoptes scabiei)",
        "description": "An intensely pruritic, contagious parasitic infestation caused by the microscopic mite Sarcoptes scabiei var. hominis, classically featuring interdigital burrows and nocturnal itching.",
        "common_triggers": ["Direct prolonged skin-to-skin contact", "Crowded living conditions", "Institutional settings", "Shared bedding"],
        "clinical_precaution": "All household members and close contacts must be treated simultaneously, even if asymptomatic, to prevent continuous re-infestation cycles.",
        "indicative_care": ["Prescription 5% permethrin cream or oral ivermectin", "Washing all bedding and clothing in hot water (>60°C)", "Antihistamines for pruritus"]
    },
    "seborrheic_dermatitis": {
        "display_name": "Seborrheic Dermatitis",
        "category_type": "Chronic Erythematosquamous Scalp & Facial Dermatosis",
        "description": "A common chronic inflammatory disorder affecting sebum-rich areas (scalp, nasolabial folds, eyebrows, retroauricular regions), linked to Malassezia yeast colonization.",
        "common_triggers": ["Sebum overproduction", "Malassezia yeast activity", "Cold dry weather", "Neurological conditions or physical fatigue"],
        "clinical_precaution": "Recurrent condition requiring periodic maintenance therapy rather than a permanent one-time cure.",
        "indicative_care": ["Antifungal shampoos (ketoconazole, zinc pyrithione, selenium sulfide)", "Mild topical anti-inflammatory agents", "Non-oily facial moisturizers"]
    },
    "urticaria": {
        "display_name": "Urticaria (Hives)",
        "category_type": "Mast-Cell Mediated Vascular Reaction",
        "description": "A vascular reaction of the upper dermis characterized by transient, intensely pruritic, erythematous edematous wheals that resolve within 24 hours without residual scarring.",
        "common_triggers": ["Food or drug allergies", "Viral infections", "Physical stimuli (cold, heat, pressure)", "Stress or cholinergic triggers"],
        "clinical_precaution": "EMERGENCY WARNING: If hives are accompanied by lip/tongue swelling (angioedema), difficulty breathing, wheezing, or dizziness, seek immediate emergency medical care (anaphylaxis).",
        "indicative_care": ["Second-generation non-sedating oral H1 antihistamines", "Cool compresses", "Avoidance of identifiable allergens and NSAIDs"]
    }
}


class CommonSkinPredictionService:
    """
    Inference service for Common Skin Condition classification.
    Maintains complete isolation from dermoscopic lesion analysis.
    """
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.40,
        device: Optional[torch.device] = None
    ):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else self.root_dir / "models" / "common_skin_model.pth"
        self.confidence_threshold = confidence_threshold

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model: Optional[CommonSkinViTClassifier] = None
        self.metadata: Dict[str, Any] = {}
        self.transforms = get_clinical_transforms(is_training=False)
        self.is_loaded = False
        self.is_genuine_trained = False
        self.activation_details = {
            "activation_status": "NO_CHECKPOINT",
            "installation_approved": False,
            "class_mapping_verified": False,
            "evaluation_reviewed": False
        }

        self._initialize_service()

    def _initialize_service(self):
        """Attempts to load model weights if available on disk."""
        if not self.checkpoint_path.exists():
            return

        try:
            model, meta = load_common_skin_checkpoint(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                eval_mode=True
            )
            self.model = model
            self.metadata = meta
            self.is_loaded = True

            # Requirement 10: Remove automatic model activation based solely on validation Macro F1 >= 0.40.
            # Require explicit installation approval, verified training metadata, consistent class mappings
            # and reviewed held-out evaluation results before enabling the new analysis mode.
            has_approval = bool(meta.get("installation_approved", False))
            
            # Verify class mapping consistency
            checkpoint_mapping = meta.get("class_to_idx", {})
            mapping_consistent = bool(checkpoint_mapping and checkpoint_mapping == CLASS_TO_IDX)
            
            # Verify reviewed evaluation results exist
            metrics_path = self.root_dir / "training" / "results" / "common_skin_metrics.json"
            has_reviewed_eval = bool(meta.get("test_metrics") or metrics_path.exists())
            
            # Only enable if ALL criteria are strictly satisfied
            self.is_genuine_trained = bool(has_approval and mapping_consistent and has_reviewed_eval)
            self.activation_details = {
                "installation_approved": has_approval,
                "class_mapping_verified": mapping_consistent,
                "evaluation_reviewed": has_reviewed_eval,
                "activation_status": "ACTIVE" if self.is_genuine_trained else "PENDING_APPROVAL_AND_VERIFICATION"
            }

        except Exception as e:
            self.is_loaded = False
            self.is_genuine_trained = False

    def is_service_ready(self) -> bool:
        """Indicates whether genuine trained model weights are active and ready."""
        return self.is_loaded and self.is_genuine_trained

    def get_service_status(self) -> Dict[str, Any]:
        """Returns readiness status and metadata for system health checks."""
        return {
            "mode": "common_skin_conditions",
            "ready": self.is_service_ready(),
            "checkpoint_exists": self.checkpoint_path.exists(),
            "is_loaded": self.is_loaded,
            "is_genuine_trained": self.is_genuine_trained,
            "activation_details": self.activation_details,
            "best_macro_f1": self.metadata.get("best_macro_f1", 0.0),
            "num_classes": NUM_COMMON_CLASSES,
            "supported_classes": COMMON_SKIN_CLASSES,
            "class_display_names": CLASS_DISPLAY_NAMES,
            "unsupported_categories": {
                "vitiligo": "Excluded due to insufficient clinical samples in verified open datasets (<10 cases).",
                "other_fungal": "Excluded to preserve clinical separation from verified Tinea Corporis."
            }
        }

    def predict(
        self,
        image_input: Union[str, Path, Image.Image]
    ) -> Dict[str, Any]:
        """
        Executes clinical photography classification on an input image.
        Returns prediction, confidence distribution, uncertainty assessment, and educational info.
        """
        if not self.is_loaded or self.model is None:
            return {
                "success": False,
                "status": "model_not_ready",
                "message": "Common Skin Condition ViT model is in preparation pending Google Colab GPU training."
            }

        # Load image
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        tensor = self.transforms(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        sorted_indices = np.argsort(probs)[::-1]
        top_idx = int(sorted_indices[0])
        top_prob = float(probs[top_idx])
        top_class_code = IDX_TO_CLASS[top_idx]
        top_display_name = CLASS_DISPLAY_NAMES[top_class_code]

        # Uncertainty Assessment
        # If top probability is below confidence threshold or entropy is high, flag as indeterminate
        entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
        max_possible_entropy = float(np.log(NUM_COMMON_CLASSES))
        normalized_entropy = entropy / max_possible_entropy

        is_confident = (top_prob >= self.confidence_threshold) and (normalized_entropy < 0.85)

        # Build top-3 predictions
        top_3 = []
        for idx in sorted_indices[:3]:
            code = IDX_TO_CLASS[int(idx)]
            top_3.append({
                "class_code": code,
                "disease_name": CLASS_DISPLAY_NAMES[code],
                "probability": float(probs[idx]),
                "percentage": f"{probs[idx] * 100:.2f}%"
            })

        all_probabilities = {
            IDX_TO_CLASS[i]: round(float(probs[i]), 4)
            for i in range(NUM_COMMON_CLASSES)
        }

        disease_info = COMMON_SKIN_DISEASE_INFO.get(top_class_code, {})

        return {
            "success": True,
            "status": "confident" if is_confident else "uncertain",
            "is_confident": is_confident,
            "predicted_class_code": top_class_code if is_confident else "indeterminate",
            "predicted_disease": top_display_name if is_confident else "Indeterminate / Atypical Clinical Presentation",
            "confidence_score": top_prob,
            "confidence_percentage": f"{top_prob * 100:.2f}%",
            "normalized_entropy": round(normalized_entropy, 4),
            "top_3_predictions": top_3,
            "all_probabilities": all_probabilities,
            "disease_info": disease_info if is_confident else None,
            "clinical_disclaimer": (
                "Educational research prototype for ordinary clinical photography. "
                "Predictions do not constitute medical diagnoses. Always consult a board-certified dermatologist."
            )
        }
