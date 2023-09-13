def get_model(model_name, args):
    name = model_name.lower()
    if name == "icarl":
        from models.icarl import iCaRL
        return iCaRL(args)
    elif name == "ewc":
        from models.ewc import EWC
        return EWC(args)
    elif name == "der":
        from models.der import DER
        return DER(args)
    elif name == "der_2stage":
        from models.der_2stage import DER_2stage
        return DER_2stage(args)
    elif name == "icarl_2stage":
        from models.icarl_2stage import iCaRL_2stage
        return iCaRL_2stage(args)
    elif name == "ewc_2stage":
        from models.ewc_2stage import EWC_2stage
        return EWC_2stage(args)
    
    else:
        assert 0
