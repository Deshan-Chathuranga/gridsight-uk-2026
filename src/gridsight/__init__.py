import sys
import importlib

def _register_module_aliases():
    aliases = {
        "lstm_q": "gridsight.models.lstm",
        "chronos_q": "gridsight.models.chronos",
        "modeling": "gridsight.models.stacking",
        "data_ingestion": "gridsight.data",
    }
    
    # Pre-defined known submodules to map
    submodules = {
        "lstm_q": ["predict", "config", "data", "train", "evaluate", "tune", "base", "base.lstm_q"],
        "chronos_q": ["predict", "config", "data", "forecast", "evaluate", "base", "base.chronos_q"],
        "modeling": ["predict", "config", "data", "train", "evaluate", "stacking", "cli", "base", "base.lgbm_q", "base.tcn_q"],
        "data_ingestion": [
            "bronze", "silver", "gold", "sync_bronze",
            "bronze.common", "bronze.cli", "bronze.fetch_met_office_live",
            "bronze.met_office", "bronze.neso", "bronze.ocf_pv", "bronze.pv_live", "bronze.upload",
            "silver.common", "silver.cli", "silver.contracts", "silver.clean",
            "gold.contracts", "gold.lag_features", "gold.merge", "gold.cli"
        ]
    }
    
    for alias, target in aliases.items():
        try:
            # Import target and alias it
            target_mod = importlib.import_module(target)
            sys.modules[alias] = target_mod
            
            # Map all its submodules
            for sub in submodules.get(alias, []):
                try:
                    sub_target = f"{target}.{sub}"
                    sub_alias = f"{alias}.{sub}"
                    sub_mod = importlib.import_module(sub_target)
                    sys.modules[sub_alias] = sub_mod
                except ImportError:
                    pass
        except ImportError:
            pass

# Execute alias registration
_register_module_aliases()
