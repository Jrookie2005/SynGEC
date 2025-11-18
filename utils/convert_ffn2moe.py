import torch
import sys
import os

# Add the path to the custom fairseq and src
sys.path.insert(0, '/home/jr/research/SeMGEC/src/src_syngec/fairseq-0.10.2')
sys.path.insert(0, '/home/jr/research/SeMGEC/src')

print("Python path:", sys.path[:3])  # Print first 3 paths for debugging

from fairseq.data import Dictionary, LabelDictionary
from src_syngec.syngec_model.syntax_guided_gec_task import SyntaxEnhancedTranslationTask
import importlib

# Dynamically import the modules
bart_module = importlib.import_module('src_syngec.syngec_model.syntax_enhanced_bart')
bart_moe_module = importlib.import_module('src_syngec.syngec_model.syntax_enhanced_bart_moe')

SyntaxEnhancedBARTModel = bart_module.SyntaxEnhancedBARTModel
SyntaxEnhancedBARTMoeModel = bart_moe_module.SyntaxEnhancedBARTMoeModel

pt_path = '/home/jr/research/SeMGEC/model/syngec/emnlp2022_syngec_chinese_bart_syngec.pt'
moe_pt_path = '/home/jr/research/SeMGEC/model/syngec/syngec_chinese_bart_moe_baseline.pt'

def convert_ffn_to_moe():
    # Load the original model directly to GPU to save memory
    print("Loading original model...")
    # Use torch.load directly with GPU map_location to avoid CPU memory usage
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint directly to GPU
    state = torch.load(pt_path, map_location=device)
    args = state['args']
    print("Args attributes:", dir(args))  # 或 print(vars(args))

    # Create the original model to get the structure
    from fairseq.data import Dictionary
    src_dict = Dictionary()
    tgt_dict = Dictionary()
    
    # Load dictionaries if they exist
    dict_path = 'preprocess/chinese_hsk+lang8_with_syntax_transformer/bin/dict.src.txt'
    if os.path.exists(dict_path):
        src_dict = Dictionary.load(dict_path)
        tgt_dict = src_dict  # For BART, src and tgt dicts are the same
    label_dict_path = 'preprocess/chinese_hsk+lang8_with_syntax_transformer/bin/dict.label.txt'
    if os.path.exists(label_dict_path):
        syntax_label_dict = []
        syntax_label_dict.append(LabelDictionary.load(label_dict_path)) 
    
    print(f"Original dict size: {len(src_dict)}")
    print(f"Args dict size: {getattr(args, 'dict_size', 'N/A')}")
    print(f"Args encoder_embed_dim: {args.encoder_embed_dim}")
    print(f"Args decoder_embed_dim: {args.decoder_embed_dim}")
    
    # Check if checkpoint has different dict size
    if hasattr(state['model'], 'keys'):
        embed_keys = [k for k in state['model'].keys() if 'embed_tokens.weight' in k]
        if embed_keys:
            checkpoint_vocab_size = state['model'][embed_keys[0]].shape[0]
            print(f"Checkpoint vocab size: {checkpoint_vocab_size}")
            if checkpoint_vocab_size != len(src_dict):
                print(f"Warning: Checkpoint vocab size ({checkpoint_vocab_size}) != dict size ({len(src_dict)})")
                # Create a dictionary that matches the checkpoint vocab size
                src_dict = Dictionary()
                # Manually set the indices to match checkpoint
                src_dict.indices = {}
                src_dict.symbols = []
                for i in range(checkpoint_vocab_size):
                    symbol = f"token_{i}"
                    src_dict.symbols.append(symbol)
                    src_dict.indices[symbol] = i
                src_dict.nspecial = 0  # No special tokens
                tgt_dict = src_dict
                print(f"Created dict with size: {len(src_dict)}")
    
    # Create a dummy task for build_model
    from fairseq.tasks import TASK_REGISTRY
    from fairseq.tasks.fairseq_task import FairseqTask
    
    
    # 考虑使用真实task，避免不可知问题
    class DummyTask(FairseqTask):
        def __init__(self, src_dict, tgt_dict, label_dict=None):
            super().__init__(None)
            self.src_dict = src_dict
            self.tgt_dict = tgt_dict
            self.syntax_label_dict = label_dict
        
        @property
        def source_dictionary(self):
            return self.src_dict
        
        @property
        def target_dictionary(self):
            return self.tgt_dict
    
    
    
    # Set MoE parameters for the MoE model BEFORE creating it
    args.moe_num_experts = 4  # Set number of experts
    args.use_moe_decoder = True  # Enable MoE decoder
    args.use_syntax = True
    args.only_gnn = True
    args.syntax_encoder = "GCN"
    args.syntax_type = ["dep"]
    
    task = SyntaxEnhancedTranslationTask(args, src_dict, tgt_dict, syntax_label_dict)
    
    # Create the MoE model with the MoE parameters set
    print("Creating MoE model...")
    moe_model = SyntaxEnhancedBARTMoeModel.build_model(args, task)
    
    # Get the state dict of the original model (from the loaded checkpoint)
    original_state_dict = state['model']

    # Clear the loaded state to free memory
    del state
    import gc
    gc.collect()    # Create a new state dict for the MoE model
    moe_state_dict = {}

    # Copy shared parameters (keep on CPU)
    print("Copying shared parameters...")
    copied_keys = []
    skipped_keys = []
    for key, value in original_state_dict.items():
        if 'decoder.layers' not in key or 'fc' not in key:
            moe_state_dict[key] = value
            copied_keys.append(key)
        else:
            # Handle decoder layer FFN to MoE conversion
            if 'fc1' in key or 'fc2' in key:
                # Skip FFN parameters as they will be replaced by MoE experts
                skipped_keys.append(key)
                continue
            else:
                moe_state_dict[key] = value
                copied_keys.append(key)
    
    print(f"Copied {len(copied_keys)} parameters, skipped {len(skipped_keys)} FFN parameters")
    print("Skipped FFN keys:", [k for k in skipped_keys if 'decoder.layers.0' in k])

    # Now handle the MoE specific parameters
    # For each decoder layer, we need to initialize the MoE experts
    num_decoder_layers = args.decoder_layers
    embed_dim = args.decoder_embed_dim
    ffn_embed_dim = args.decoder_ffn_embed_dim

    print(f"Converting {num_decoder_layers} decoder layers...")

    # Process layers one by one to save memory
    for layer_idx in range(num_decoder_layers):
        print(f"Processing layer {layer_idx + 1}/{num_decoder_layers}")
        
        # Get the original FFN weights
        fc1_weight = original_state_dict[f'decoder.layers.{layer_idx}.fc1.weight']
        fc1_bias = original_state_dict[f'decoder.layers.{layer_idx}.fc1.bias']
        fc2_weight = original_state_dict[f'decoder.layers.{layer_idx}.fc2.weight']
        fc2_bias = original_state_dict[f'decoder.layers.{layer_idx}.fc2.bias']

        # For MoE, we assume 4 experts (can be adjusted)
        num_experts = getattr(args, 'moe_num_experts', 4)

        # Split the FFN parameters across experts
        # For simplicity, replicate the same weights to all experts
        # In practice, you might want to use different initialization or expert specialization

        for expert_idx in range(num_experts):
            # Expert weights - replicate the original FFN weights
            moe_state_dict[f'decoder.layers.{layer_idx}.moe_ffn.experts.{expert_idx}.0.weight'] = fc1_weight.clone()
            moe_state_dict[f'decoder.layers.{layer_idx}.moe_ffn.experts.{expert_idx}.0.bias'] = fc1_bias.clone()
            moe_state_dict[f'decoder.layers.{layer_idx}.moe_ffn.experts.{expert_idx}.3.weight'] = fc2_weight.clone()
            moe_state_dict[f'decoder.layers.{layer_idx}.moe_ffn.experts.{expert_idx}.3.bias'] = fc2_bias.clone()

        # Initialize gate parameters (simple linear layer)
        gate_weight = torch.randn(num_experts, embed_dim)
        gate_bias = torch.zeros(num_experts)

        moe_state_dict[f'decoder.layers.{layer_idx}.moe_ffn.gate.weight'] = gate_weight
        moe_state_dict[f'decoder.layers.{layer_idx}.moe_ffn.gate.bias'] = gate_bias

    # Clear original model and state dict to free memory
    del original_state_dict
    gc.collect()

    # Load the converted state dict into the MoE model
    print("Loading converted state dict...")
    moe_model.load_state_dict(moe_state_dict, strict=False)

    # Clear the state dict
    del moe_state_dict
    gc.collect()

    # Save the new model (move back to CPU for saving)
    print("Saving MoE model...")
    moe_model_cpu = moe_model.cpu()  # Move back to CPU for saving
    torch.save({
        'model': moe_model_cpu.state_dict(),
        'args': args,
        'epoch': 0,
        'best_loss': None,
    }, moe_pt_path)

    print(f"Conversion completed. MoE model saved to {moe_pt_path}")

def analyze_model_sizes():
    """Analyze and compare model sizes"""
    import torch
    
    print("=== Model Size Analysis ===")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device}")
    # Load baseline model
    print("Loading baseline model...")
    baseline_state = torch.load(pt_path, map_location=device)
    baseline_model_params = baseline_state['model']
    
    # Load MoE model  
    print("Loading MoE model...")
    moe_state = torch.load(moe_pt_path, map_location=device)
    moe_model_params = moe_state['model']
    
    def count_parameters(state_dict):
        total_params = 0
        param_counts = {}
        for key, param in state_dict.items():
            param_count = param.numel()
            total_params += param_count
            # Group by layer type
            if 'decoder.layers' in key:
                layer_type = key.split('decoder.layers.')[1].split('.')[0]
                if layer_type not in param_counts:
                    param_counts[layer_type] = 0
                param_counts[layer_type] += param_count
            else:
                if 'shared' not in param_counts:
                    param_counts['shared'] = 0
                param_counts['shared'] += param_count
        return total_params, param_counts
    
    baseline_total, baseline_counts = count_parameters(baseline_model_params)
    moe_total, moe_counts = count_parameters(moe_model_params)
    
    print(f"\nBaseline model total parameters: {baseline_total:,}")
    print(f"MoE model total parameters: {moe_total:,}")
    print(f"Ratio: {moe_total/baseline_total:.2f}")
    
    print("\nBaseline parameter breakdown:")
    for k, v in baseline_counts.items():
        print(f"  {k}: {v:,}")
    
    print("\nMoE parameter breakdown:")
    for k, v in moe_counts.items():
        print(f"  {k}: {v:,}")
    
    # Check specific layer parameters
    print("\n=== Detailed Layer Analysis ===")
    baseline_layer_keys = [k for k in baseline_model_params.keys() if 'decoder.layers.0' in k]
    moe_layer_keys = [k for k in moe_model_params.keys() if 'decoder.layers.0' in k or 'decoder.layers.2' in k]
    
    print("Baseline layer 0 keys:")
    for k in sorted(baseline_layer_keys):
        shape = baseline_model_params[k].shape
        params = baseline_model_params[k].numel()
        print(f"  {k}: {shape} = {params:,}")
    
    print("\nMoE layer 0 keys:")
    for k in sorted(moe_layer_keys):
        shape = moe_model_params[k].shape
        params = moe_model_params[k].numel()
        print(f"  {k}: {shape} = {params:,}")

if __name__ == "__main__":
    import sys
    print(sys.argv)
    if len(sys.argv) > 1 or sys.argv[1] == 'analyze':
        analyze_model_sizes()
    else:
        convert_ffn_to_moe()