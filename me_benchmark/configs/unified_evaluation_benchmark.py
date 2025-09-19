"""
Configuration for unified evaluation benchmark
"""
from typing import Dict, Any, List

# Default configuration for unified evaluation
DEFAULT_UNIFIED_EVAL_CONFIG = {
    "evaluators": [
        {
            "type": "easyedit",
            "name": "knowledge_editing_evaluator",
            "metrics": [
                "rewrite_acc",      # Reliability
                "rephrase_acc",     # Generalization
                "locality",         # Locality
                "portability"       # Portability
            ],
            "additional_metrics": [
                "fluency",          # Generation quality
                "consistency"       # Concept consistency (for ConceptEdit)
            ],
            "hparams": {
                "device": 0,
                "alg_name": "ROME"
            }
        },
        {
            "type": "opencompass",
            "name": "language_understanding_evaluator",
            "datasets": [
                {
                    "name": "mmlu",
                    "subset": "all",
                    "metrics": ["accuracy"]
                },
                {
                    "name": "hellaswag",
                    "metrics": ["accuracy"]
                },
                {
                    "name": "winogrande",
                    "metrics": ["accuracy"]
                }
            ],
            "metrics": ["accuracy"]
        }
    ],
    "datasets": [
        {
            "type": "unified",
            "name": "comprehensive_benchmark",
            "sources": [
                {
                    "type": "easyedit",
                    "name": "zsre",
                    "path": "data/zsre/zsre_mend_eval.json",
                    "kwargs": {
                        "dataset_type": "zsre"
                    }
                },
                {
                    "type": "easyedit",
                    "name": "counterfact",
                    "path": "data/counterfact/counterfact-edit.json",
                    "kwargs": {
                        "dataset_type": "counterfact"
                    }
                },
                {
                    "type": "knowedit",
                    "name": "knowedit",
                    "path": "data/knowedit/ZsRE-test-all.json",
                    "kwargs": {
                        "dataset_type": "knowedit"
                    }
                }
            ]
        }
    ],
    "metrics": {
        "knowledge_editing": {
            "reliability": {
                "description": "Edit success rate on target instances",
                "higher_is_better": True
            },
            "generalization": {
                "description": "Edit success rate on rephrased instances",
                "higher_is_better": True
            },
            "locality": {
                "description": "Preservation of performance on unrelated instances",
                "higher_is_better": True
            },
            "portability": {
                "description": "Edit success rate on related reasoning tasks",
                "higher_is_better": True
            },
            "fluency": {
                "description": "Quality of generated text after editing",
                "higher_is_better": True
            }
        },
        "language_understanding": {
            "mmlu": {
                "description": "Massive Multitask Language Understanding",
                "higher_is_better": True
            },
            "hellaswag": {
                "description": "Reasoning about event sequences",
                "higher_is_better": True
            },
            "winogrande": {
                "description": "Commonsense reasoning",
                "higher_is_better": True
            }
        }
    },
    "benchmark_protocol": {
        "pre_edit_evaluation": True,
        "post_edit_evaluation": True,
        "sequential_editing": False,
        "batch_editing": False,
        "evaluation_mode": "comprehensive"  # comprehensive, fast, detailed
    }
}


# Configuration for different benchmark scenarios
BENCHMARK_SCENARIOS = {
    "knowledge_editing_only": {
        "evaluators": [
            {
                "type": "easyedit",
                "metrics": ["rewrite_acc", "rephrase_acc", "locality", "portability", "fluency"]
            }
        ],
        "datasets": [
            {
                "type": "easyedit",
                "name": "zsre",
                "path": "data/zsre/zsre_mend_eval.json"
            }
        ]
    },
    
    "language_understanding_only": {
        "evaluators": [
            {
                "type": "opencompass",
                "datasets": ["mmlu", "hellaswag", "winogrande"]
            }
        ],
        "datasets": [
            {
                "type": "opencompass",
                "name": "mmlu",
                "subset": "all"
            }
        ]
    },
    
    "comprehensive": DEFAULT_UNIFIED_EVAL_CONFIG,
    
    "fast_evaluation": {
        "evaluators": [
            {
                "type": "easyedit",
                "metrics": ["rewrite_acc"]
            },
            {
                "type": "opencompass",
                "datasets": ["mmlu"]
            }
        ],
        "datasets": [
            {
                "type": "unified",
                "sources": [
                    {
                        "type": "easyedit",
                        "name": "zsre_small",
                        "path": "data/zsre/zsre_mend_eval_small.json"
                    },
                    {
                        "type": "opencompass",
                        "name": "mmlu_small",
                        "subset": "stem"
                    }
                ]
            }
        ],
        "benchmark_protocol": {
            "pre_edit_evaluation": False,
            "post_edit_evaluation": True,
            "evaluation_mode": "fast"
        }
    }
}


def get_benchmark_config(scenario: str = "comprehensive") -> Dict[str, Any]:
    """Get benchmark configuration for a specific scenario"""
    return BENCHMARK_SCENARIOS.get(scenario, DEFAULT_UNIFIED_EVAL_CONFIG)


def customize_benchmark_config(base_scenario: str, customizations: Dict[str, Any]) -> Dict[str, Any]:
    """Customize benchmark configuration"""
    config = get_benchmark_config(base_scenario).copy()
    
    # Apply customizations
    for key, value in customizations.items():
        if key in config:
            if isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
        else:
            config[key] = value
            
    return config