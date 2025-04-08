set -e
set -u

ref_audio=$1
text_path=$2

echo "ref_audio: ${ref_audio}"
echo "text_path: ${text_path}"

export PYTHONPATH=$(pwd)
echo $(pwd)

text=`cat ${text_path}`

# 获取 token
python fish_speech/models/vqgan/inference.py \
    -i ${ref_audio} \
    --checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"


python fish_speech/models/text2semantic/inference.py \
    --text "${text}" \
    --prompt-text "The text corresponding to reference audio" \
    --prompt-tokens "fake.npy" \
    --checkpoint-path "checkpoints/fish-speech-1.5" \
    --num-samples 2
    # --compile

python fish_speech/models/vqgan/inference.py \
    -i "temp/codes_0.npy" \
    --checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
