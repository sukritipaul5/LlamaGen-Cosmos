import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import os
import time
import sys
sys.path.append('/fs/cml-projects/yet-another-diffusion/LlamaGen')


import argparse
from tokenizer.tokenizer_image.vq_model import VQ_models
from language.t5 import T5Embedder
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torchao
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import clip
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F


def calculate_clip_score(images, prompts, device):
    """Calculate CLIP score between images and prompts"""
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Prepare images
    clip_transform = transforms.Compose([
        transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]),  # [-1,1] to [0,1]
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    processed_images = torch.stack([clip_transform(img) for img in images])
    
    # Encode images and text
    with torch.no_grad():
        image_features = model.encode_image(processed_images)
        text_features = model.encode_text(clip.tokenize(prompts).to(device))
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity
        similarity = (100.0 * image_features @ text_features.T).diagonal()
        
    return similarity.mean().item(), similarity.tolist()

def calculate_lpips(generated_images, reference_images=None):
    """Calculate LPIPS score between generated images or against reference images"""
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(generated_images.device)
    
    # Normalize images to [-1, 1] range if they aren't already
    def normalize_images(imgs):
        # Check if normalization is needed
        min_val, max_val = imgs.min(), imgs.max()
        if min_val < -1.0 or max_val > 1.0:
            imgs = torch.clamp(imgs, -1.0, 1.0)
        return imgs
    
    if reference_images is None:
        # Calculate LPIPS between consecutive pairs of generated images
        scores = []
        normalized_images = normalize_images(generated_images)
        
        for i in range(0, len(normalized_images)-1, 2):
            score = lpips(normalized_images[i:i+1], normalized_images[i+1:i+2])
            scores.append(score.item())
        return np.mean(scores), scores
    else:
        # Calculate LPIPS between generated and reference images
        scores = []
        normalized_gen = normalize_images(generated_images)
        normalized_ref = normalize_images(reference_images)
        
        for gen_img, ref_img in zip(normalized_gen, normalized_ref):
            score = lpips(gen_img.unsqueeze(0), ref_img.unsqueeze(0))
            scores.append(score.item())
        return np.mean(scores), scores

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"image tokenizer is loaded")

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        block_size=latent_size ** 2,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
 
    if "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint


    print(f"gpt model is loaded")

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 
    
    assert os.path.exists(args.t5_path)
    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )
    # general prompts
    prompt_type = "general"
    prompts = [
      "Perched on a windowsill, a black cat gazes out into the garden.",
      "Over a calm river in autumn, a wooden bridge stretches peacefully.",
      "Steam rises from a cup of coffee sitting beside an open book on the table.",
      "At a lively carnival, a cheerful clown holds a cluster of balloons.",
      "Driving through green fields, a farmer navigates their sturdy tractor.",
      "In a sleek modern kitchen, a robot gracefully serves tea.",
      "Bright umbrellas and intricate sandcastles scatter across a sunny beach.",
      "On the table lies a fruit bowl brimming with apples, bananas, and grapes.",
      "Pedaling along a forest trail, a cyclist enjoys the morning air.",
      "Above a futuristic city, a flying car hovers with smooth precision.",
      "Two children laugh as they splash water from brightly colored water guns.",
      "Resting on a soft carpet, a golden retriever chews a favorite toy.",
      "In a vast field, the vibrant yellow petals of a sunflower catch the sunlight.",
      "A pizza bakes to perfection inside a traditional wood-fired oven.",
      "Sliding effortlessly across the ice, a penguin enjoys its Antarctic home.",
      "Cakes, pies, and pastries fill a bakery counter, ready to delight customers.",
      "After the rain, a rainbow arches across a serene green valley.",
      "Floating high in the sky, a colorful hot air balloon drifts with the wind.",
      "Left on a mountain trail, a pair of hiking boots tells a hiker’s story.",
      "Logs encircle a cozy campfire, glowing warmly under a starry sky.",
      "Through the countryside, a train cuts through the landscape as dusk falls.",
      "Holding an ice cream cone, a child smiles brightly under the summer sun.",
      "Docked quietly on a tranquil lake, a small boat awaits its next journey.",
      "Neon lights glow outside a coffee shop, cutting through the night’s darkness.",
      "Pancakes flip high in the air as a chef works in a bustling kitchen.",
      "In a shallow pond, flamingos stand gracefully with reflections in the water.",
      "Suspended in space, an astronaut gazes at the vibrant blue of Earth.",
      "Under the shade of blooming cherry blossoms, a park bench invites rest.",
      "Golden syrup glistens on a stack of pancakes topped with a pat of butter.",
      "Amid colorful plants in an aquarium, a small fish swims peacefully.",
      "Tall and sturdy, a lighthouse stands watch over the rocky shoreline.",
      "In a glass vase, a bouquet of roses adds charm to the dining table.",
      "Stretching its neck, a giraffe nibbles leaves from the top of a tree.",
      "With a carrot nose and a scarf, a snowman smiles in the winter yard.",
      "A bustling town square features a fountain splashing gently at its center.",
      "Parked in front of a retro diner, a red vintage car catches the eye.",
      "Bubbles drift on the breeze as a little girl giggles and blows more.",
      "At the lake’s edge, an artist captures the beauty of the scene on canvas.",
      "Under the shade of an oak tree, a family enjoys a picnic together.",
      "Schools of fish dart through the ocean, weaving between coral reefs.",
      "Waves lap at their feet as a young couple strolls along the beach.",
      "Beneath a maple tree, fallen autumn leaves form a golden carpet.",
      "Vibrant vegetables and fresh flowers fill the stalls at a farmer’s market.",
      "In a sunny meadow, a playful puppy chases butterflies with boundless energy."
    ]
    # Hard prompts
    # prompt_type = "hard"
    # prompts =  [
    # "A cat sitting on a dog's back, both under a wooden table.",
    # "Three apples stacked on top of each other next to a vase with sunflowers.",
    # "A child holding a balloon while standing on a seesaw with a bird perched on the other end.",
    # "A chair inside a room reflected perfectly in a cracked mirror on the wall.",
    # "Two identical bicycles parked side by side, one leaning against a tree, the other under the tree.",
    # "Ten red balloons floating in the sky with five green balloons tied to a tree.",
    # "Seven distinct books of varying sizes stacked beside a cup of tea on a wooden table.",
    # "Four black cats sitting in a row on a white bench, each looking in a different direction.",
    # "Three blue cars parked next to two yellow trucks on a busy city street.",
    # "Fifteen brightly colored flowers blooming in a garden, with five bees flying among them.",
    # "A street sign that reads 'Welcome to Dreamland' in bold, neon letters.",
    # "A billboard in Times Square with the text 'The Future is Now' clearly displayed.",
    # "A handwritten recipe on a notebook page with legible ingredients and steps.",
    # "A coffee mug with the phrase 'Best Dad Ever' printed in a fun font.",
    # "A book cover featuring the title 'Mysteries of the Universe' and an author's name.",
    # "A person holding a cup of coffee, showing five fingers wrapped naturally around the handle.",
    # "Two dancers with outstretched arms and interlocking fingers in a graceful pose.",
    # "A close-up of a person's face, smiling with evenly aligned teeth and natural lips.",
    # "A group of people high-fiving, each hand distinctly visible and fully formed.",
    # "A yoga instructor in a one-legged pose, hands in prayer position and body balanced.",
    # "A middle-aged woman with freckles and curly red hair, wearing glasses and smiling.",
    # "A young man with dark skin and textured hair, lit by soft sunlight in a park.",
    # "A child with East Asian features, wearing a bright yellow raincoat and holding an umbrella.",
    # "An elderly man with a white beard and warm brown eyes, sitting under a tree.",
    # "A group portrait of friends from diverse ethnicities, all laughing with natural expressions.",
    # "A microscopic view of bacteria swimming in a drop of water, showing detailed cell shapes.",
    # "An aerial view of a cityscape at night, with visible cars and people in the streets below.",
    # "A single drop of dew on a blade of grass, reflecting the surrounding forest in its surface.",
    # "A galaxy seen from a distance, showing billions of stars in detailed spiral arms.",
    # "A close-up of an ant carrying a crumb, with the surrounding environment in soft focus.",
    # "A mountain range during sunrise with mist rolling over the peaks and valleys.",
    # "A crystal-clear lake reflecting the sky and surrounding pine trees on a calm day.",
    # "A desert with sand dunes stretching to the horizon, lit by a dramatic sunset.",
    # "A dense rainforest with sunlight filtering through the canopy and a waterfall in the distance.",
    # "A vast icy tundra under the Northern Lights, with distant snow-capped mountains.",
    # "A minimalist poster for a jazz concert, featuring a saxophone and bold retro typography.",
    # "An event flyer for a tech conference with clean lines, futuristic fonts, and holographic effects.",
    # "A movie poster for a sci-fi thriller, showcasing a mysterious planet and dramatic lighting.",
    # "A colorful slide for a children's presentation, featuring playful fonts and animal illustrations.",
    # "An abstract art poster for a gallery opening, with vibrant colors and geometric patterns.",
    # "A cheerful anime character with lavender hair, wearing a school uniform, standing in a cherry blossom park.",
    # "A cartoon hero in a futuristic cityscape, with robotic sidekicks and glowing skyscrapers.",
    # "A fantasy anime character holding a glowing sword, standing on a cliff overlooking a mystical valley.",
    # "A whimsical cartoon character cooking in a kitchen full of oversized utensils and animated food.",
    # "A group of anime friends sitting around a campfire at night, with stars twinkling above them."
    # ]

    caption_embs, emb_masks = t5_model.get_text_embeddings(prompts)


    if not args.no_left_padding:
        print(f"processing left-padding...")    
        # a naive way to implement left-padding
        new_emb_masks = torch.flip(emb_masks, dims=[-1])
        new_caption_embs = []
        for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
            valid_num = int(emb_mask.sum().item())
            print(f'  prompt {idx} token len: {valid_num}')
            new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
            new_caption_embs.append(new_caption_emb)
        new_caption_embs = torch.stack(new_caption_embs)
    else:
        new_caption_embs, new_emb_masks = caption_embs, emb_masks
    c_indices = new_caption_embs * new_emb_masks[:,:, None]
    c_emb_masks = new_emb_masks

    qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]
    t1 = time.time()
    index_sample = generate(
        gpt_model, c_indices, latent_size ** 2, 
        c_emb_masks, 
        cfg_scale=args.cfg_scale,
        temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, sample_logits=True, 
        )
    sampling_time = time.time() - t1
    print(f"Full sampling takes about {sampling_time:.2f} seconds.")    
    
    t2 = time.time()
    samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")

    os.makedirs("./outputs/seperate", exist_ok=True)
    
    # Convert the batch of tensors to seperate images
    for idx, (sample, prompt) in enumerate(zip(samples, prompts)):
        # Save the generated image
        save_image(sample, f"./outputs/seperate/image_{idx:03d}.png", normalize=True, value_range=(-1, 1))
        
        # Create a new image with text
        img = Image.open(f"./outputs/seperate/image_{idx:03d}.png")
        # Create a new image with padding for text
        target_width = img.width
        target_height = img.height + 150  # Add 100 pixels for text
        
        combined = Image.new('RGB', (target_width, target_height), 'white')
        combined.paste(img, (0, 0))
        
        # Add text
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            
        # Wrap text if too long
        words = prompt.split()
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            if len(' '.join(current_line)) * 10 > target_width - 20:  # Approximate width
                lines.append(' '.join(current_line[:-1]))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
            
        # Draw each line
        y_text = img.height + 10
        for line in lines:
            draw.text((10, y_text), line, fill='black', font=font)
            y_text += 20
            
        combined.save(f"./outputs/seperate/image_{idx:03d}_with_prompt.png")
    
    # Keep the original combined grid save as well
    save_image(samples, f"./outputs/sample_{prompt_type}_{args.gpt_type}.png", nrow=4, normalize=True, value_range=(-1, 1))
    print(f"seperate images saved to ./outputs/seperate/")
    print(f"Combined grid saved to ./outputs/sample_{prompt_type}_{args.gpt_type}.png")

    # After generating and saving images, add evaluation code:
    print("\nCalculating evaluation metrics...")
    
    # Create evaluation results directory
    eval_dir = "./outputs/evaluation"
    os.makedirs(eval_dir, exist_ok=True)
    
    eval_results = []
    
    # Calculate CLIP score
    clip_score_mean, clip_scores = calculate_clip_score(samples, prompts, device)
    eval_results.append(f"CLIP Score (mean): {clip_score_mean:.4f}")
    eval_results.append("Individual CLIP Scores:")
    for i, score in enumerate(clip_scores):
        eval_results.append(f"  Image {i}: {score:.4f}")
    print(f"CLIP Score (mean): {clip_score_mean:.4f}")
    
    # Calculate LPIPS score
    lpips_mean, lpips_scores = calculate_lpips(samples)
    eval_results.append(f"LPIPS Score (mean): {lpips_mean:.4f}")
    eval_results.append("Individual LPIPS Scores:")
    for i, score in enumerate(lpips_scores):
        eval_results.append(f"  Pair {i}: {score:.4f}")
    print(f"LPIPS Score (mean): {lpips_mean:.4f}")
    
    # Save evaluation results
    with open(os.path.join(eval_dir, f"evaluation_results_{prompt_type}_{args.gpt_type}.txt"), "w") as f:
        f.write("\n".join(eval_results))
    
    print(f"Evaluation results saved to {eval_dir}/evaluation_results_{prompt_type}_{args.gpt_type}.txt")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5-path", type=str, default='pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--no-left-padding", action='store_true', default=False)
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i", help="class->image or text->image")  
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--real-images-dir", type=str, default=None, 
                      help="Directory containing real images for FID calculation")
    args = parser.parse_args()
    main(args)
