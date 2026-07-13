import argparse
import os
import wandb
from datetime import datetime as dt
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.nn.functional as F
import torchvision.utils as tvu

from models.tarflow import TarFlowRaw
from models.push_forward import DriftModelRaw
from models.memory_bank import TorchMemoryBank
from models.drift_loss import NFDriftLossFunction_LearnableRatio_Regularize
from utils.seed import seed_everything
from utils.fid import *
from utils.wandb import get_best_checkpoint, get_wandb_run_info, find_project_run_dirctory
from utils.lr import get_lr_schedule, set_lr
from data.two_dimensional import get_2d_dataset, save_2d_dataset_image, save_2d_arrow_diagram


RANDOM_SEED = 46


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--generative_model_path', default='weights/', type=str, help='Model weight directory')
    parser.add_argument('--output_dir', default='results/', type=str, help='Output directory')
    
    parser.add_argument('--dataset_name', default="spiral", choices=["spiral"], help="Dataset name")
    parser.add_argument('--img_size', default=8, type=int, help="Output image size")
    parser.add_argument('--channel_size', default=1, type=int, help="Channel size")
    
    # parser.add_argument('--num_push_forward_blocks', default=8, type=int, help="Number of FlowBlocks")
    parser.add_argument('--num_flow_blocks', default=8, type=int, help="Number of FlowBlocks")
    parser.add_argument('--flow_block_dim', default=1024, type=int, help="Internal dim of FlowBlocks")
    parser.add_argument('--permutation_type', default="flip", choices=["flip", "shuffle"], help="Type of permutation for the NF")
    parser.add_argument('--num_attn_blocks', default=8, type=int, help="Number of Attention blocks per FlowBlocks")
    # parser.add_argument('--hidden_dim', default=512, type=int, help="Hidden dimension of the entire model")
    parser.add_argument('--attn_num_heads', default=64, type=int, help="Head dim of Attention blocks")
    parser.add_argument('--attn_head_dim', default=64, type=int, help="Head dim of Attention blocks")
    parser.add_argument('--attn_temp', default=1.0, type=float, help='Attention temperature')
    parser.add_argument('--ffn_expansion', default=4, type=int, help="Internal dim multiplier for FFN layer")
    parser.add_argument('--cfg_weight', default=0.0, type=float, help='Guidance weight for sampling, 0 is no guidance')
    parser.add_argument("--annealed_guidance", action="store_true", help="Apply the annealed guidance")
    parser.add_argument('--init_vp_vq_ratio', default=1.0, type=float, help='Initial ratio V_p/V_q')
    parser.add_argument('--reg_lambda', default=0.01, type=float, help='Parameter for the Regularization loss on V_q')

    parser.add_argument('--batch_size', default=128, type=int, help='Training batch size across all devices')
    # parser.add_argument('--positive_sample_ratio', default=0.5, type=float, help='pos/(pos+neg) sample ratio')
    parser.add_argument('--epochs', default=100, type=int, help='Training epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Maximum learning rate')
    parser.add_argument('--lr_schedule_type', default='wsd', type=str, choices=["wsd", "ws", "d", "cos", "s"], help='Learning rate schedule')
    # parser.add_argument('--class_dropout_prob', default=0, type=float, help='Ratio for random label drop in conditional mode')
    parser.add_argument('--sample_freq', default=1, type=int, help='Frequency of sampling in terms of epochs')
    # parser.add_argument('--num_samples', default=4096, type=int, help='Number of sampels to draw')
    # parser.add_argument('--sample_batch_size', default=256, type=int, help='Batch size for drawing samples')
    parser.add_argument("--dry_run", action="store_true", help="Simple test run in local.")
    parser.add_argument('--resume_wandb_url', default='', type=str, help='URL at wandb')    
    parser.add_argument(
        '--compile', default=False, action=argparse.BooleanOptionalAction, help='Whether to use torch.compile, expect the first epoch to be slow when enabled'
    )
    
    args = parser.parse_args()
    # assert args.num_samples >= args.sample_batch_size, f"args.num_samples={args.num_samples} less than args.sample_batch_size(={args.sample_batch_size})."
    
    file_name = os.path.basename(__file__).split(".")[0]
    now_str = dt.now().strftime("%y%m%d_%H%M")    
    exp_config_dict = {}
    print(f'{" Config ":-^80}')
    for k, v in sorted(vars(args).items()):
        exp_config_dict[k] = v
        print(f'{k:32s}: {v}')    
        
    
    settings_str = ""
    settings_str += f"init_vp_vq_ratio_{args.init_vp_vq_ratio}-reg_lambda_{args.reg_lambda}"
    # settings_str += f"-n_flowblck_{args.num_flow_blocks}-n_attnblck_{args.num_attn_blocks}"
    # settings_str += f"-n_attnheads_{args.attn_num_heads}-attnhead_dim_{args.attn_head_dim}"
    settings_str += f"-lr_{args.lr}-batch_sz_{args.batch_size}"
        
    wandb_exp_name = f"DriftingNF-{file_name}-{args.dataset_name}"
    wandb_run_name = settings_str
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(RANDOM_SEED)
    
    # 2-Dim Unconditional Case
    num_classes = 1     # Spiral and Uniform
    num_patches = 2
    
    # model = DriftModelRaw(
    #     in_channels=args.channel_size,
    #     num_patches=num_patches,
    #     num_blocks=args.num_push_forward_blocks,
    #     hidden_dim=args.hidden_dim,
    #     attn_num_heads=args.attn_num_heads,
    #     ffn_expansion=args.ffn_expansion,
    # ).to(device)    
    
    # Detect autograd anomaly
    torch.autograd.set_detect_anomaly(True)
    
    model = TarFlowRaw(
        in_channels = args.channel_size,
        num_patches = num_patches,     # 2-dimensional
        num_flow_blocks = args.num_flow_blocks,
        flow_block_dim = args.flow_block_dim,
        attn_num_heads = args.attn_num_heads,
        num_attn_blocks = args.num_attn_blocks,
        permutation_type = args.permutation_type,
        attn_head_dim = args.attn_head_dim,
        # attn_temp= args.attn_temp,
        ffn_expansion = args.ffn_expansion,
        num_classes = num_classes,
        # class_dropout_prob=args.class_dropout_prob,
    ).to(device)
    
    # Drift Loss Implemented as an independent class
    nf_drift_loss_model = NFDriftLossFunction_LearnableRatio_Regularize(
        init_vp_vq_ratio=args.init_vp_vq_ratio, reg_lambda=args.reg_lambda
    )
        
    all_params = list(model.parameters()) + list(nf_drift_loss_model.parameters()) # Learnable V_p-V_q ratio added
    num_parameters = sum(p.numel() for p in all_params if p.requires_grad)
    print(f"Number of parameters: {num_parameters}, {num_parameters / 1e6}M")
        
    optimizer = torch.optim.AdamW(all_params, betas=(0.9, 0.95), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler()
    
    # Memory Banks for Samples
    pos_bank_size, neg_bank_size = args.batch_size * 2, args.batch_size * 8
    memory_bank_positive = TorchMemoryBank(num_classes, max_size=pos_bank_size)
    # memory_bank_negative = TorchMemoryBank(1, max_size=neg_bank_size)
    pos_per_sample = 32
    # neg_per_sample = 16
    gen_per_label = 16

    # Resume related settings
    start_epoch = 0
    min_train_loss = torch.inf
    # min_valid_loss = torch.inf    
    
    # Directory settings
    base_weights_dir = os.path.join(args.generative_model_path, f"train_weights_wip/{file_name}")    
    base_weights_dir = os.path.join(base_weights_dir, args.dataset_name)
    base_results_dir = os.path.join(args.output_dir, f"train/{file_name}")
    base_results_dir = os.path.join(base_results_dir, args.dataset_name)
    
    if args.dry_run:
        base_weights_dir = os.path.join(base_weights_dir, f"dry_run-{now_str}-{settings_str}")
        base_results_dir = os.path.join(base_results_dir, f"dry_run-{now_str}-{settings_str}")
            
    else:
        if args.resume_wandb_url and args.resume_wandb_url.lower() != "false":  # In case of bash passing false
            user_name, project_name, run_id = get_wandb_run_info(args.resume_wandb_url)
            wandb.init(
                project=project_name,
                entity=user_name, 
                id=run_id,
                resume="must",
            )
            
            print(f"user_name, project_name, run_id : {user_name}, {project_name}, {run_id}")
            base_weights_dir = find_project_run_dirctory(base_weights_dir, run_id, 2)
            base_results_dir = find_project_run_dirctory(base_results_dir, run_id, 2)
            print(f"found base_weights_dir : {base_weights_dir}")
            print(f"found base_results_dir : {base_results_dir}")
            
            # base_weights_dir = os.path.join(base_weights_dir, f"wandb-{now_str}-{run_id}-{settings_str}")
            # base_results_dir = os.path.join(base_results_dir, f"wandb-{now_str}-{run_id}-{settings_str}")
                
            print(f"Resume training.")
            
            ws_weights_dir = os.path.join(base_weights_dir, f"lr_schedule_type_ws")
            assert os.path.exists(ws_weights_dir), f"weights_dir {ws_weights_dir} does not exist for resuming."
            
            print(f"=> Getting checkpoints from: {ws_weights_dir}")
            last_checkpoint = get_best_checkpoint(ws_weights_dir)
            checkpoint = torch.load(last_checkpoint, weights_only=False)
            start_epoch = checkpoint['epoch'] + 1
            
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"=> Loaded model weights.")
            
            if 'min_train_loss' in checkpoint: 
                min_train_loss = checkpoint['min_train_loss']
            
            # if 'min_valid_loss' in checkpoint: 
            #     min_valid_loss = checkpoint['min_valid_loss']
            
            print(f"=> Resuming from epoch {start_epoch}")
                
        else:
            run_id = os.environ.get("WANDB_RUN_ID", wandb.util.generate_id())
            print(f"Initiate wandb with new run_id : {run_id}")
            
            wandb.init(
                project=wandb_exp_name, 
                name=wandb_run_name,
                id=run_id,
                resume="allow",
                config = exp_config_dict,
            )
                
            base_weights_dir = os.path.join(base_weights_dir, f"wandb-{now_str}-{run_id}-{settings_str}")
            base_results_dir = os.path.join(base_results_dir, f"wandb-{now_str}-{run_id}-{settings_str}")
    
    # Directory settings based on wandb run_id
    weights_dir = os.path.join(base_weights_dir, f"lr_schedule_type_{args.lr_schedule_type}")
    results_dir = os.path.join(base_results_dir, f"lr_schedule_type_{args.lr_schedule_type}")    
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Training loop settings
    save_weight_freq = args.epochs // 10
    patience_cnt = 0
    tolerance_cnt = 50
    
    fixed_noise_eval = torch.randn((args.batch_size, num_patches, 1), device=device)
    fixed_y = None
    # fixed_y = torch.randint(num_classes, (args.num_samples,))

    # fid = FID(reset_real_features=False, normalize=True).to(device)
    # fid_stat_dir = get_fid_stats_dir(args.dataset_name, args.img_size)
    # if os.path.exists(fid_stat_dir):
    #     print(f'Loading FID stats from {fid_stat_dir}')
    #     fid.load_state_dict(torch.load(fid_stat_dir, map_location='cpu', weights_only=False))
    # else:
    #     prepare_fid_stats(ds_train, args.dataset_name, args.img_size, args.batch_size, fid_stat_dir)
    
    # def compute_loss(x, y, device, model):
    #     with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
    #         z, outputs, log_dets = model(x, y)
    #         loss = model.get_loss(z, log_dets)
    #         return loss, (z, outputs, log_dets)
    
    # print(f"[torch.compile] : ", end="")
    # if args.compile:
    #     compute_loss = torch.compile(compute_loss, fullgraph=False, backend='inductor', mode='max-autotune')
    # print(f"fin.")
    
    
    for epoch in tqdm(range(start_epoch, start_epoch + args.epochs)):
        curr_lr = get_lr_schedule(epoch, start_epoch + args.epochs, args.lr, args.lr_schedule_type, start_epoch=start_epoch)
        set_lr(optimizer, curr_lr)
        
        # Push samples to memory banks for training
        # Consider Two label cases for now : Spiral and Uniform
        spiral_samples = get_2d_dataset(n_points=args.batch_size, dataset_name=args.dataset_name)
        spiral_labels = torch.zeros((args.batch_size, ), dtype=torch.long)
        memory_bank_positive.add(spiral_samples, spiral_labels)
        
        # uniform_samples = get_2d_dataset(n_points=args.batch_size, dataset_name="uniform")
        # uniform_labels = torch.zeros((args.batch_size, ), dtype=torch.long)
        # memory_bank_negative.add(uniform_samples, uniform_labels)
        # # Incorporate both pos/neg to neg-memory-bank
        # negative_samples = torch.cat([spiral_samples, uniform_samples], dim=0)
        # negative_labels = torch.cat([spiral_labels, uniform_labels], dim=0)
        # memory_bank_negative.add(negative_samples, negative_labels)
        
        # if epoch == start_epoch:
        #     save_2d_dataset_image(spiral_samples, args.img_size, results_dir, f"training_data-epoch_{epoch+1}")
        
        # train mode
        model.train()
        
        # Sample from Banks
        labels_for_sample = torch.zeros((args.batch_size, ), dtype=torch.long)
        pos_samples = memory_bank_positive.sample(labels_for_sample, num_samples=pos_per_sample)    # (B, N_pos, ...)
        # neg_samples = memory_bank_negative.sample(labels_for_sample*0, num_samples=neg_per_sample)  # (B, N_neg, ...)
        
        # Stop grad
        pos_samples = pos_samples.reshape(args.batch_size, pos_per_sample, -1).to(device)
        # neg_samples = neg_samples.reshape(args.batch_size, neg_per_sample, -1).detach().to(device)
        # neg_samples = None
        
        # Changed logic of sampling everything at once
        train_noise = torch.randn((args.batch_size * gen_per_label, num_patches, 1), device=device)
        x_gen = model.reverse(train_noise, y=None)
        
        x_gen_detached = x_gen.detach().requires_grad_(True)
        
        with sdpa_kernel([SDPBackend.MATH]):        
            z, outputs, log_dets = model(x_gen_detached, y=None)
            nf_loss = 0.5 * z.pow(2).mean(dim=(-1, -2)) - log_dets      # (B*N_gen, )
            log_likelihood = -nf_loss
            nf_score = torch.autograd.grad(
                outputs=log_likelihood.sum(),
                inputs=x_gen_detached,
                create_graph=True,     # For the reg_loss
                retain_graph=True,     # For the reg_loss
            )[0]
        nf_score = nf_score.reshape(args.batch_size, gen_per_label, -1) # (B, N_gen, ...)

        
        x_gen = x_gen.reshape(args.batch_size, gen_per_label, -1)       # (B, N_gen, ...)
        # z = z.reshape(args.batch_size, gen_per_label, -1)               # (B, N_gen, ...)
        
        # z_for_loss = z.reshape(args.batch_size, gen_per_label, -1).detach().clone()
        # nf_score_for_loss = nf_score.detach().clone()
                
        optimizer.zero_grad()
        # curr_loss = nf_drift_loss_model(z, pos_samples, nf_score)   # (B, )
        curr_loss = nf_drift_loss_model(x_gen, pos_samples, nf_score)   # (B, )
        curr_loss = curr_loss.mean()
        scaler.scale(curr_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        raw_learned_ratio = nf_drift_loss_model.inverted_softplus_ratio
        actual_learned_ratio = F.softplus(raw_learned_ratio).item()  
        
        if not args.dry_run:
            wandb.log({
                "train_loss": curr_loss.item(),
                "V_p-V_q-ratio": actual_learned_ratio,
                "learning_rate": curr_lr,
                "grad_norm": grad_norm.item(),
                "epoch": epoch,
            })        
        
        curr_loss = curr_loss.item()
        if not args.dry_run and curr_loss < min_train_loss:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'min_train_loss' : min_train_loss,
            }
            torch.save(checkpoint, os.path.join(weights_dir, f"model_best.pth"))            
        
        draw_flag = False
        if curr_loss < min_train_loss:
            draw_flag = True
        min_train_loss = min(min_train_loss, curr_loss)
        # del pos_samples
        
        # Evaluate performance
        if (epoch + 1) % args.sample_freq == 0:
            model.eval()
            noise_eval = fixed_noise_eval.clone().to(device)
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    gen_samples_eval = model.reverse(noise_eval, y=None).detach()

            # fid_score = fid.compute().item()
            # fid.reset()
            
            if (epoch + 1) % (args.sample_freq * 5) == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'min_train_loss' : min_train_loss,
                }
                torch.save(checkpoint, os.path.join(weights_dir, f"epoch_{epoch+1}-loss_{min_train_loss:.2f}.pth"))
            
            gen_samples_eval = gen_samples_eval.reshape(args.batch_size, -1).unsqueeze(1)    # (B, 1, S) where C_gen=1

            v_p, _ = nf_drift_loss_model._calculate_positive_force(
                gen_samples_eval, pos_samples, 1e-6, nf_drift_loss_model.temperature_list
            )
            v_p_np = v_p.reshape(args.batch_size, -1).detach().cpu().numpy()
            
            gen_samples_clone = gen_samples_eval.clone().reshape(args.batch_size, num_patches, num_classes).requires_grad_(True)
            z_q, _, log_dets_q = model(gen_samples_clone, y=None)
            nf_loss_q = 0.5 * z_q.pow(2).mean(dim=(-1, -2)) - log_dets_q    # (B*N_gen, )
            log_likelihood_q = -nf_loss_q
            nf_score_q = torch.autograd.grad(
                outputs=log_likelihood_q.sum(),
                inputs=gen_samples_clone,
                create_graph=False,
                retain_graph=False,
            )[0]
            nf_score_q = nf_score_q.reshape(args.batch_size, -1) # C_gen=1
            v_q_np = nf_score_q.detach().cpu().numpy()
            
            # Refer to Model ratio for visualization
            model_raw_ratio = nf_drift_loss_model.inverted_softplus_ratio
            model_actual_ratio = F.softplus(model_raw_ratio).item()
            ratio_total = model_actual_ratio + 1.0
            weight_p = 2.0 * (model_actual_ratio / ratio_total)
            weight_q = 2.0 * (1.0 / ratio_total)
            v_p_np = v_p_np * weight_p
            v_q_np = v_q_np * weight_q            
            
            pos_samples_np = pos_samples.view(args.batch_size * pos_per_sample, -1)
            pos_samples_np = pos_samples_np.unsqueeze(-1).float().cpu().numpy()
            gen_samples_np = gen_samples_eval.squeeze(1).unsqueeze(-1)
            gen_samples_np = gen_samples_np.float().cpu().numpy()
            # save_2d_dataset_image(generated_samples_eval, args.img_size, results_dir, f"inference-epoch_{epoch+1}")
            
            save_2d_arrow_diagram(pos_samples_np, gen_samples_np, v_p_np, v_q_np, epoch, results_dir, f"inference-epoch_{epoch+1}")            
            
            
            
    
    if not args.dry_run:
        wandb.finish()
        
    exit()