# Train all versions of DiT - Polyps
#python train.py --model-name DiT_S8_polyps --model DiT_S8 --data-path ./data/polyps --epochs 150 --cross-model false
#python train.py --model-name DiT_S4_polyps --model DiT_S4 --data-path ./data/polyps --epochs 150 --cross-model false
#python train.py --model-name DiT_S2_polyps --model DiT_S2 --data-path ./data/polyps --epochs 150 --cross-model false
#python train.py --model-name DiT_B8_polyps --model DiT_B8 --data-path ./data/polyps --epochs 150 --cross-model false
#python train.py --model-name DiT_B4_polyps --model DiT_B4 --data-path ./data/polyps --epochs 150 --cross-model false
#python train.py --model-name DiT_B2_polyps --model DiT_B2 --data-path ./data/polyps --epochs 150 --cross-model false
#python train.py --model-name DiT_L2_polyps --model DiT_L2 --data-path ./data/polyps --epochs 150 --cross-model false



# Train all versions of DiT cross - Polyps
#python train.py --model-name DiT_S8_CROSS_polyps --model DiT_S8 --data-path ./data/polyps --epochs 150 --cross-model true
#python train.py --model-name DiT_S4_CROSS_polyps --model DiT_S4 --data-path ./data/polyps --epochs 150 --cross-model true
#python train.py --model-name DiT_S2_CROSS_polyps --model DiT_S2 --data-path ./data/polyps --epochs 150 --cross-model true
#python train.py --model-name DiT_B8_CROSS_polyps --model DiT_B8 --data-path ./data/polyps --epochs 150 --cross-model true
#python train.py --model-name DiT_B4_CROSS_polyps --model DiT_B4 --data-path ./data/polyps --epochs 150 --cross-model true
#python train.py --model-name DiT_B2_CROSS_polyps --model DiT_B2 --data-path ./data/polyps --epochs 150 --cross-model true


# Train all versions of DiT - Kvasir
#python train.py --model-name DiT_S8_Kvasir --model DiT_S8 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model false
#python train.py --model-name DiT_S4_Kvasir --model DiT_S4 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model false
#python train.py --model-name DiT_S2_Kvasir --model DiT_S2 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model false
#python train.py --model-name DiT_B8_Kvasir --model DiT_B8 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model false
#python train.py --model-name DiT_B4_Kvasir --model DiT_B4 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model false
#python train.py --model-name DiT_B2_Kvasir --model DiT_B2 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model false

# Train all versions of DiT cross - Kvasir
#python train.py --model-name DiT_S8_CROSS_Kvasir --model DiT_S8 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model true
#python train.py --model-name DiT_S4_CROSS_Kvasir --model DiT_S4 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model true
#python train.py --model-name DiT_S2_CROSS_Kvasir --model DiT_S2 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model true
#python train.py --model-name DiT_B8_CROSS_Kvasir --model DiT_B8 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model true
#python train.py --model-name DiT_B4_CROSS_Kvasir --model DiT_B4 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model true
#python train.py --model-name DiT_B2_CROSS_Kvasir --model DiT_B2 --data-path ./data/Kvasir-SEG --epochs 150 --cross-model true

# Sample
#python sample.py --model-name DiT_S8_CROSS_polyps --model DiT_S8 --data-path ./data/polyps --cross-model true --ema true --prediction-path ./saved_models/DiT_S8_CROSS_polyps/ema-pred --num-images 20
#python sample.py --model-name DiT_S4_CROSS_polyps --model DiT_S4 --data-path ./data/polyps --cross-model true --ema true --prediction-path ./saved_models/DiT_S4_CROSS_polyps/ema-pred --num-images 20
#python sample.py --model-name DiT_S2_CROSS_polyps --model DiT_S2 --data-path ./data/polyps --cross-model true --ema true --prediction-path ./saved_models/DiT_S2_CROSS_polyps/ema-pred --num-images 20
#
#python sample.py --model-name DiT_S8_CROSS_polyps --model DiT_S8 --data-path ./data/polyps --cross-model true --ema false --prediction-path ./saved_models/DiT_S8_CROSS_polyps/pred --num-images 20
#python sample.py --model-name DiT_S4_CROSS_polyps --model DiT_S4 --data-path ./data/polyps --cross-model true --ema false --prediction-path ./saved_models/DiT_S4_CROSS_polyps/pred --num-images 20
#python sample.py --model-name DiT_S2_CROSS_polyps --model DiT_S2 --data-path ./data/polyps --cross-model true --ema false --prediction-path ./saved_models/DiT_S2_CROSS_polyps/pred --num-images 20
#
#python sample.py --model-name DiT_S8_polyps --model DiT_S8 --data-path ./data/polyps --cross-model false --ema true --prediction-path ./saved_models/DiT_S8_polyps/ema-pred --num-images 20
#python sample.py --model-name DiT_S4_polyps --model DiT_S4 --data-path ./data/polyps --cross-model false --ema true --prediction-path ./saved_models/DiT_S4_polyps/ema-pred --num-images 20
#python sample.py --model-name DiT_S2_polyps --model DiT_S2 --data-path ./data/polyps --cross-model false --ema true --prediction-path ./saved_models/DiT_S2_polyps/ema-pred --num-images 20
#
#python sample.py --model-name DiT_S8_polyps --model DiT_S8 --data-path ./data/polyps --cross-model false --ema false --prediction-path ./saved_models/DiT_S8_polyps/pred --num-images 20
#python sample.py --model-name DiT_S4_polyps --model DiT_S4 --data-path ./data/polyps --cross-model false --ema false --prediction-path ./saved_models/DiT_S4_polyps/pred --num-images 20
#python sample.py --model-name DiT_S2_polyps --model DiT_S2 --data-path ./data/polyps --cross-model false --ema false --prediction-path ./saved_models/DiT_S2_polyps/pred --num-images 20
#
#
#python sample.py --model-name DiT_S8_CROSS_Kvasir --model DiT_S8 --data-path ./data/Kvasir-SEG --cross-model true --ema true --prediction-path ./saved_models/DiT_S8_CROSS_Kvasir/ema-pred --num-images 20
#python sample.py --model-name DiT_S4_CROSS_Kvasir --model DiT_S4 --data-path ./data/Kvasir-SEG --cross-model true --ema true --prediction-path ./saved_models/DiT_S4_CROSS_Kvasir/ema-pred --num-images 20
#python sample.py --model-name DiT_S2_CROSS_Kvasir --model DiT_S2 --data-path ./data/Kvasir-SEG --cross-model true --ema true --prediction-path ./saved_models/DiT_S2_CROSS_Kvasir/ema-pred --num-images 20
#
#python sample.py --model-name DiT_S8_CROSS_Kvasir --model DiT_S8 --data-path ./data/Kvasir-SEG --cross-model true --ema false --prediction-path ./saved_models/DiT_S8_CROSS_Kvasir/pred --num-images 20
#python sample.py --model-name DiT_S4_CROSS_Kvasir --model DiT_S4 --data-path ./data/Kvasir-SEG --cross-model true --ema false --prediction-path ./saved_models/DiT_S4_CROSS_Kvasir/pred --num-images 20
#python sample.py --model-name DiT_S2_CROSS_Kvasir --model DiT_S2 --data-path ./data/Kvasir-SEG --cross-model true --ema false --prediction-path ./saved_models/DiT_S2_CROSS_Kvasir/pred --num-images 20
#
#python sample.py --model-name DiT_S8_Kvasir --model DiT_S8 --data-path ./data/Kvasir-SEG --cross-model false --ema true --prediction-path ./saved_models/DiT_S8_Kvasir/ema-pred --num-images 20
#python sample.py --model-name DiT_S4_Kvasir --model DiT_S4 --data-path ./data/Kvasir-SEG --cross-model false --ema true --prediction-path ./saved_models/DiT_S4_Kvasir/ema-pred --num-images 20
#python sample.py --model-name DiT_S2_Kvasir --model DiT_S2 --data-path ./data/Kvasir-SEG --cross-model false --ema true --prediction-path ./saved_models/DiT_S2_Kvasir/ema-pred --num-images 20
#
#python sample.py --model-name DiT_S8_Kvasir --model DiT_S8 --data-path ./data/Kvasir-SEG --cross-model false --ema false --prediction-path ./saved_models/DiT_S8_Kvasir/pred --num-images 20
#python sample.py --model-name DiT_S4_Kvasir --model DiT_S4 --data-path ./data/Kvasir-SEG --cross-model false --ema false --prediction-path ./saved_models/DiT_S4_Kvasir/pred --num-images 20
#python sample.py --model-name DiT_S2_Kvasir --model DiT_S2 --data-path ./data/Kvasir-SEG --cross-model false --ema false --prediction-path ./saved_models/DiT_S2_Kvasir/pred --num-images 20
#
#
#python sample.py --model-name DiT_B8_CROSS_polyps --model DiT_B8 --data-path ./data/Kvasir-SEG --cross-model true --ema true --prediction-path ./saved_models/DiT_B8_CROSS_Kvasir/ema-pred --num-images 20
#python sample.py --model-name DiT_B4_CROSS_polyps --model DiT_B4 --data-path ./data/Kvasir-SEG --cross-model true --ema true --prediction-path ./saved_models/DiT_B4_CROSS_Kvasir/ema-pred --num-images 20
#python sample.py --model-name DiT_B2_CROSS_polyps --model DiT_B2 --data-path ./data/Kvasir-SEG --cross-model true --ema true --prediction-path ./saved_models/DiT_B2_CROSS_Kvasir/ema-pred --num-images 20
#
#python sample.py --model-name DiT_B8_CROSS_polyps --model DiT_B8 --data-path ./data/Kvasir-SEG --cross-model true --ema false --prediction-path ./saved_models/DiT_B8_CROSS_Kvasir/pred --num-images 20
#python sample.py --model-name DiT_B4_CROSS_polyps --model DiT_B4 --data-path ./data/Kvasir-SEG --cross-model true --ema false --prediction-path ./saved_models/DiT_B4_CROSS_Kvasir/pred --num-images 20
#python sample.py --model-name DiT_B2_CROSS_polyps --model DiT_B2 --data-path ./data/Kvasir-SEG --cross-model true --ema false --prediction-path ./saved_models/DiT_B2_CROSS_Kvasir/pred --num-images 20
#
#python sample.py --model-name DiT_B8_polyps --model DiT_B8 --data-path ./data/polyps --cross-model false --ema true --prediction-path ./saved_models/DiT_B8_polyps/ema-pred --num-images 20
#python sample.py --model-name DiT_B4_polyps --model DiT_B4 --data-path ./data/polyps --cross-model false --ema true --prediction-path ./saved_models/DiT_B4_polyps/ema-pred --num-images 20
#python sample.py --model-name DiT_B2_polyps --model DiT_B2 --data-path ./data/polyps --cross-model false --ema true --prediction-path ./saved_models/DiT_B2_polyps/ema-pred --num-images 20
#
#python sample.py --model-name DiT_B8_polyps --model DiT_B8 --data-path ./data/polyps --cross-model false --ema false --prediction-path ./saved_models/DiT_B8_polyps/pred --num-images 20
#python sample.py --model-name DiT_B4_polyps --model DiT_B4 --data-path ./data/polyps --cross-model false --ema false --prediction-path ./saved_models/DiT_B4_polyps/pred --num-images 20
#python sample.py --model-name DiT_B2_polyps --model DiT_B2 --data-path ./data/polyps --cross-model false --ema false --prediction-path ./saved_models/DiT_B2_polyps/pred --num-images 20
#
#
#python sample.py --model-name DiT_B8_CROSS_Kvasir --model DiT_B8 --data-path ./data/Kvasir-SEG --cross-model true --ema true --prediction-path ./saved_models/DiT_B8_CROSS_Kvasir/ema-pred --num-images 20
#python sample.py --model-name DiT_B4_CROSS_Kvasir --model DiT_B4 --data-path ./data/Kvasir-SEG --cross-model true --ema true --prediction-path ./saved_models/DiT_B4_CROSS_Kvasir/ema-pred --num-images 20
#python sample.py --model-name DiT_B2_CROSS_Kvasir --model DiT_B2 --data-path ./data/Kvasir-SEG --cross-model true --ema true --prediction-path ./saved_models/DiT_B2_CROSS_Kvasir/ema-pred --num-images 20
#
#python sample.py --model-name DiT_B8_CROSS_Kvasir --model DiT_B8 --data-path ./data/Kvasir-SEG --cross-model true --ema false --prediction-path ./saved_models/DiT_B8_CROSS_Kvasir/pred --num-images 20
#python sample.py --model-name DiT_B4_CROSS_Kvasir --model DiT_B4 --data-path ./data/Kvasir-SEG --cross-model true --ema false --prediction-path ./saved_models/DiT_B4_CROSS_Kvasir/pred --num-images 20
#python sample.py --model-name DiT_B2_CROSS_Kvasir --model DiT_B2 --data-path ./data/Kvasir-SEG --cross-model true --ema false --prediction-path ./saved_models/DiT_B2_CROSS_Kvasir/pred --num-images 20
#
#python sample.py --model-name DiT_B8_Kvasir --model DiT_B8 --data-path ./data/Kvasir-SEG --cross-model false --ema true --prediction-path ./saved_models/DiT_B8_Kvasir/ema-pred --num-images 20
#python sample.py --model-name DiT_B4_Kvasir --model DiT_B4 --data-path ./data/Kvasir-SEG --cross-model false --ema true --prediction-path ./saved_models/DiT_B4_Kvasir/ema-pred --num-images 20
#python sample.py --model-name DiT_B2_Kvasir --model DiT_B2 --data-path ./data/Kvasir-SEG --cross-model false --ema true --prediction-path ./saved_models/DiT_B2_Kvasir/ema-pred --num-images 20
#
#python sample.py --model-name DiT_B8_Kvasir --model DiT_B8 --data-path ./data/Kvasir-SEG --cross-model false --ema false --prediction-path ./saved_models/DiT_B8_Kvasir/pred --num-images 20
#python sample.py --model-name DiT_B4_Kvasir --model DiT_B4 --data-path ./data/Kvasir-SEG --cross-model false --ema false --prediction-path ./saved_models/DiT_B4_Kvasir/pred --num-images 20
#python sample.py --model-name DiT_B2_Kvasir --model DiT_B2 --data-path ./data/Kvasir-SEG --cross-model false --ema false --prediction-path ./saved_models/DiT_B2_Kvasir/pred --num-images 20
#

#Models Hila is training

#only one augmentation at a time
#python train.py --model-name DiT_B2_Kvasir --model DiT_B2 --data-path ./data/kvasir-SEG --epochs 150 --cross-model false
#python sample.py --model-name DiT_B2_Kvasir --model DiT_B2 --data-path ./data/kvasir-SEG --cross-model false --ema true --prediction-path ./saved_models/DiT_B2_Kvasir/ema-pred --num-images 200
#python sample.py --model-name DiT_B2_Kvasir --model DiT_B2 --data-path ./data/kvasir-SEG --cross-model false --ema false --prediction-path ./saved_models/DiT_B2_Kvasir/pred --num-images 200
#python .\segmentation_env.py --model-name DiT_B2_Kvasir --pred-path ./saved_models/DiT_B2_Kvasir/pred --data-path ./data/Kvasir-SEG --ema false
#python .\segmentation_env.py --model-name DiT_B2_Kvasir --pred-path ./saved_models/DiT_B2_Kvasir/ema-pred --data-path ./data/Kvasir-SEG --ema true

#python train.py --model-name DiT_B2_CROSS_Kvasir --model DiT_B2 --data-path ./data/kvasir-SEG --epochs 150 --cross-model true
#python sample.py --model-name DiT_B2_CROSS_Kvasir --model DiT_B2 --data-path ./data/kvasir-SEG --cross-model true --ema true --prediction-path ./saved_models/DiT_B2_CROSS_Kvasir/ema-pred --num-images 200
#python .\segmentation_env.py --model-name DiT_B2_CROSS_Kvasir --pred-path ./saved_models/DiT_B2_CROSS_Kvasir/ema-pred --data-path ./data/kvasir-SEG --ema true
#python sample.py --model-name DiT_B2_CROSS_Kvasir --model DiT_B2 --data-path ./data/kvasir-SEG --cross-model true --ema false --prediction-path ./saved_models/DiT_B2_CROSS_Kvasir/pred --num-images 200
#python .\segmentation_env.py --model-name DiT_B2_CROSS_Kvasir --pred-path ./saved_models/DiT_B2_CROSS_Kvasir/pred --data-path ./data/kvasir-SEG --ema false

#up to 3 augmentations at a time
#python train.py --model-name DiT_B2_CROSS_polyps --model DiT_B2 --data-path ./data/polyps --epochs 150 --cross-model true
#python sample.py --model-name DiT_B2_CROSS_polyps --model DiT_B2 --data-path ./data/polyps --cross-model true --ema true --prediction-path ./saved_models/DiT_B2_CROSS_polyps/ema-pred --num-images 200
#python .\segmentation_env.py --model-name DiT_B2_CROSS_polyps --pred-path ./saved_models/DiT_B2_CROSS_polyps/ema-pred --data-path ./data/polyps
#python sample.py --model-name DiT_B2_CROSS_polyps --model DiT_B2 --data-path ./data/polyps --cross-model true --ema false --prediction-path ./saved_models/DiT_B2_CROSS_polyps/pred --num-images 200
#python .\segmentation_env.py --model-name DiT_B2_CROSS_polyps --pred-path ./saved_models/DiT_B2_CROSS_polyps/pred --data-path ./data/polyps

#python train.py --model-name DiT_B2_polyps --model DiT_B2 --data-path ./data/polyps --epochs 150 --cross-model false
#python sample.py --model-name DiT_B2_polyps --model DiT_B2 --data-path ./data/polyps --cross-model false --ema true --prediction-path ./saved_models/DiT_B2_polyps/ema-pred --num-images 200
#python .\segmentation_env.py --model-name DiT_B2_polyps --pred-path ./saved_models/DiT_B2_polyps/ema-pred --data-path ./data/polyps


#python train.py --model-name DiT_B2_Kvasir --model DiT_B2 --data-path ./data/kvasir-SEG --epochs 150 --cross-model false
#python sample.py --model-name DiT_B2_Kvasir --model DiT_B2 --data-path ./data/kvasir-SEG --cross-model false --ema true --prediction-path ./saved_models/DiT_B2_Kvasir/ema-pred --num-images 200
#python sample.py --model-name DiT_B2_Kvasir --model DiT_B2 --data-path ./data/kvasir-SEG --cross-model false --ema false --prediction-path ./saved_models/DiT_B2_Kvasir/pred --num-images 200
#python .\segmentation_env.py --model-name DiT_B2_Kvasir --pred-path ./saved_models/DiT_B2_Kvasir/pred --data-path ./data/Kvasir-SEG --ema false
#python .\segmentation_env.py --model-name DiT_B2_Kvasir --pred-path ./saved_models/DiT_B2_Kvasir/ema-pred --data-path ./data/Kvasir-SEG --ema true

#python train.py --model-name DiT_B2_CROSS_Kvasir --model DiT_B2 --data-path ./data/kvasir-SEG --epochs 150 --cross-model true
#python sample.py --model-name DiT_B2_CROSS_Kvasir --model DiT_B2 --data-path ./data/kvasir-SEG --cross-model true --ema true --prediction-path ./saved_models/DiT_B2_CROSS_Kvasir/ema-pred --num-images 200
#python .\segmentation_env.py --model-name DiT_B2_CROSS_Kvasir --pred-path ./saved_models/DiT_B2_CROSS_Kvasir/ema-pred --data-path ./data/kvasir-SEG --ema true
#python sample.py --model-name DiT_B2_CROSS_Kvasir --model DiT_B2 --data-path ./data/kvasir-SEG --cross-model true --ema false --prediction-path ./saved_models/DiT_B2_CROSS_Kvasir/pred --num-images 200
#python .\segmentation_env.py --model-name DiT_B2_CROSS_Kvasir --pred-path ./saved_models/DiT_B2_CROSS_Kvasir/pred --data-path ./data/kvasir-SEG --ema false
#python train.py --model-name DiT_B4_CROSS_Kvasir --model DiT_B4 --data-path ./data/kvasir-SEG --epochs 150 --cross-model true


#python train.py --model-name DiT_B4_Kvasir --model DiT_B4 --data-path ./data/kvasir-SEG --epochs 150 --cross-model false
#python sample.py --model-name DiT_B4_Kvasir --model DiT_B4 --data-path ./data/kvasir-SEG --cross-model false --ema false --prediction-path ./saved_models/DiT_B4_Kvasir/pred --num-images 200
#python sample.py --model-name DiT_B2_CROSS_Kvasir --model DiT_B2 --data-path ./data/kvasir-SEG --cross-model true --ema true --prediction-path ./saved_models/DiT_B2_CROSS_Kvasir/ema-pred --num-images 200
#python .\segmentation_env.py --model-name DiT_B2_CROSS_Kvasir --pred-path ./saved_models/DiT_B2_CROSS_Kvasir/ema-pred --data-path ./data/kvasir-SEG --ema true
#python sample.py --model-name DiT_B2_CROSS_Kvasir --model DiT_B2 --data-path ./data/kvasir-SEG --cross-model true --ema false --prediction-path ./saved_models/DiT_B2_CROSS_Kvasir/pred --num-images 200
#python .\segmentation_env.py --model-name DiT_B2_CROSS_Kvasir --pred-path ./saved_models/DiT_B2_CROSS_Kvasir/pred --data-path ./data/kvasir-SEG --ema false
#python train.py --model-name DiT_L4_Kvasir --model DiT_L4 --data-path ./data/kvasir-SEG --epochs 150 --cross-model false
#
#
#
#python .\segmentation_env.py --model-name DiT_B4_Kvasir --pred-path ./saved_models/DiT_B4_Kvasir/pred --data-path ./data/kvasir-SEG --ema false
#python sample.py --model-name DiT_B4_Kvasir --model DiT_B4 --data-path ./data/kvasir-SEG --cross-model false --ema true --prediction-path ./saved_models/DiT_B4_Kvasir/pred --num-images 200
#python .\segmentation_env.py --model-name DiT_B4_Kvasir --pred-path ./saved_models/DiT_B4_Kvasir/pred --data-path ./data/kvasir-SEG --ema true
#python .\segmentation_env.py --model-name DiT_B4_Kvasir --pred-path ./saved_models/DiT_B4_Kvasir/pred --data-path ./data/kvasir-SEG --ema false



#python train.py --model-name DiT_B8_CROSS_Kvasir --model DiT_B8 --data-path ./data/kvasir-SEG --epochs 150 --cross-model true


#python train.py --model-name DiT_B4_polyps --model DiT_B4 --data-path ./data/polyps --epochs 150 --cross-model false
#python train.py --model-name DiT_B4_CROSS_polyps --model DiT_B4 --data-path ./data/polyps --epochs 150 --cross-model true
#python .\segmentation_env.py --model-name DiT_B4_polyp --pred-path ./saved_models/DiT_B4_polyp/pred --data-path ./data/polyps --ema false
#python sample.py --model-name DiT_B4_polyp --model DiT_B4 --data-path ./data/polyps --cross-model false --ema false --prediction-path ./saved_models/DiT_B4_polyp/pred --num-images 200
#python train.py --model-name DiT_S2_Kvasir_32_epochs --model DiT_S2 --data-path ./data/kvasir-SEG --batch-size 32 --epochs 150 --cross-model false
#python sample.py --model-name DiT_S2_Kvasir_32_epochs --model DiT_S2 --data-path ./data/kvasir-SEG --cross-model false --ema false --prediction-path ./saved_models/DiT_S2_Kvasir_32_epochs/pred --num-images 200
#python .\segmentation_env.py --model-name DiT_S2_Kvasir_32_epochs --pred-path ./saved_models/DiT_S2_Kvasir_32_epochs/pred --data-path ./data/kvasir-SEG --ema false
#
#python train.py --model-name DiT_S2_Kvasir_8_batch --model DiT_S2 --data-path ./data/kvasir-SEG --batch-size 8 --epochs 150 --cross-model false
#python sample.py --model-name DiT_S2_Kvasir_8_batch --model DiT_S2 --data-path ./data/kvasir-SEG --cross-model false --ema false --prediction-path ./saved_models/DiT_S2_Kvasir_8_batch/pred --num-images 200
#python .\segmentation_env.py --model-name DiT_S2_Kvasir_8_batch --pred-path ./saved_models/DiT_S2_Kvasir_8_batch/pred --data-path ./data/kvasir-SEG --ema false
#
#python train.py --model-name DiT_S2_Kvasir_1_batch --model DiT_S2 --data-path ./data/kvasir-SEG --batch-size 1 --epochs 150 --cross-model false
#python sample.py --model-name DiT_S2_Kvasir_1_batch --model DiT_S2 --data-path ./data/kvasir-SEG --cross-model false --ema false --prediction-path ./saved_models/DiT_S2_Kvasir_1_batch/pred --num-images 200
#python .\segmentation_env.py --model-name DiT_S2_Kvasir_1_batch --pred-path ./saved_models/DiT_S2_Kvasir_1_batch/pred --data-path ./data/kvasir-SEG --ema false
#
#python train.py --model-name DiT_B2_Kvasir_8_batch --model DiT_B2 --data-path ./data/kvasir-SEG --batch-size 8 --epochs 150 --cross-model true
#python sample.py --model-name DiT_B2_Kvasir_8_batch --model DiT_B2 --data-path ./data/kvasir-SEG --cross-model false --ema false --prediction-path ./saved_models/DiT_B2_Kvasir_8_batch/pred --num-images 200
#python .\segmentation_env.py --model-name DiT_B2_Kvasir_8_batch --pred-path ./saved_models/DiT_B2_Kvasir_8_batch/pred --data-path ./data/kvasir-SEG --ema false
#python sample.py --model-name DiT_B2_Kvasir_8_batch --model DiT_B2 --data-path ./data/kvasir-SEG --cross-model false --ema true --prediction-path ./saved_models/DiT_B2_Kvasir_8_batch/pred-ema --num-images 200 --num-testing-steps 300
#python .\segmentation_env.py --model-name DiT_B2_Kvasir_8_batch --pred-path ./saved_models/DiT_B2_Kvasir_8_batch/pred-ema --data-path ./data/kvasir-SEG --ema true
#python sample.py --model-name DiT_B2_Kvasir_8_batch --model DiT_B2 --data-path ./data/kvasir-SEG --cross-model false --ema true --prediction-path ./saved_models/DiT_B2_Kvasir_8_batch/pred-ema --num-images 200 --num-testing-steps 500
#python sample.py --model-name DiT_B2_Kvasir_8_batch --model DiT_B2 --data-path ./data/kvasir-SEG --cross-model false --ema true --prediction-path ./saved_models/DiT_B2_Kvasir_8_batch/pred-ema --num-images 200 --num-testing-steps 400
#python sample.py --model-name DiT_B2_Kvasir_8_batch --model DiT_B2 --data-path ./data/kvasir-SEG --cross-model false --ema true --prediction-path ./saved_models/DiT_B2_Kvasir_8_batch/pred-ema --num-images 200 --num-testing-steps 200
#python sample.py --model-name DiT_B2_Kvasir_8_batch --model DiT_B2 --data-path ./data/kvasir-SEG --cross-model false --ema true --prediction-path ./saved_models/DiT_B2_Kvasir_8_batch/pred-ema --num-images 200 --num-testing-steps 350
##Running now
#python sample.py --model-name DiT_L4_Kvasir --model DiT_L4 --data-path ./data/kvasir-SEG --cross-model false --ema false --prediction-path ./saved_models/DiT_L4_Kvasir/pred --num-images 200
#python .\segmentation_env.py --model-name DiT_L4_Kvasir --pred-path ./saved_models/DiT_L4_Kvasir/pred --data-path ./data/kvasir-SEG --ema false
#python sample.py --model-name DiT_L4_Kvasir --model DiT_L4 --data-path ./data/kvasir-SEG --cross-model false --ema true --prediction-path ./saved_models/DiT_L4_Kvasir/pred-ema --num-images 200
#python .\segmentation_env.py --model-name DiT_L4_Kvasir --pred-path ./saved_models/DiT_L4_Kvasir/pred-ema --data-path ./data/kvasir-SEG --ema true

#python train.py --model-name DiT_B2_CROSS_Kvasir_8_batch --model DiT_B2 --data-path ./data/kvasir-SEG --batch-size 8 --epochs 150 --cross-model true
#python sample.py --model-name DiT_B2_CROSS_Kvasir_8_batch --model DiT_B2 --data-path ./data/kvasir-SEG --cross-model true --ema true --prediction-path ./saved_models/DiT_B2_CROSS_Kvasir_8_batch/pred --num-images 200 --num-testing-steps 300
#python segmentation_env.py --model-name DiT_B2_CROSS_Kvasir_8_batch --pred-path ./saved_models/DiT_B2_CROSS_Kvasir_8_batch/pred --data-path ./data/kvasir-SEG --ema true --num-testing-steps 300

#running now
#save embedded images
#python save_embedded_images.py --data_path ./data/kvasir-SEG --images_path ./data/kvasir-SEG/images --gt_path ./data/kvasir-SEG/masks --train_fraction 0.8 --resize 256 --num-augmentations 4
#python save_embedded_images.py --data_path ./data/polyps --images_path ./data/polyps/train/train --gt_path ./data/polyps/train_gt/train_gt --train_fraction 0.8 --resize 256 --num-augmentations 4
#
##train models
#python train.py --model DiT_B2 --data-path ./data/kvasir-SEG --batch-size 8 --epochs 150 --cross-model false --num-augmentations 4
#python sample.py --model DiT_B2 --data-path ./data/kvasir-SEG --batch-size 8 --epochs 150 --cross-model false --num-augmentations 4 --ema false --num-images 200 --num-testing-steps 300
#python segmentation_env.py --model DiT_B2 --data-path ./data/kvasir-SEG --batch-size 8 --epochs 150 --cross-model false --num-augmentations 4 --ema false --num-testing-steps 300
#
#python sample.py --model DiT_B2 --data-path ./data/kvasir-SEG --batch-size 8 --epochs 150 --cross-model false --num-augmentations 4 --ema true --num-images 200 --num-testing-steps 100
#python segmentation_env.py --model DiT_B2 --data-path ./data/kvasir-SEG --batch-size 8 --epochs 150 --cross-model false --num-augmentations 4 --ema true --num-testing-steps 100
#
#python sample.py --model DiT_B2 --data-path ./data/kvasir-SEG --batch-size 8 --epochs 150 --cross-model false --num-augmentations 4 --ema true --num-images 200 --num-testing-steps 200
#python segmentation_env.py --model DiT_B2 --data-path ./data/kvasir-SEG --batch-size 8 --epochs 150 --cross-model false --num-augmentations 4 --ema true --num-testing-steps 200
#
#python sample.py --model DiT_B2 --data-path ./data/kvasir-SEG --batch-size 8 --epochs 150 --cross-model false --num-augmentations 4 --ema true --num-images 200 --num-testing-steps 300
#python segmentation_env.py --model DiT_B2 --data-path ./data/kvasir-SEG --batch-size 8 --epochs 150 --cross-model false --num-augmentations 4 --ema true --num-testing-steps 300

python sample.py --model DiT_B2 --batch-size 8 --cross-model false --num-augmentations 1 --num-images 50 --cfg 0.0
