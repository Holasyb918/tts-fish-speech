import os
reference_audio_dict = {
                        'asr_format_333.wav': '咱们只需要对着手机说一二三就可以了。一二三一二三一二三四五六七，一二三四五六七八九十，一二三四五六七八九十，一二三三二一，一二三四五六七，一二三四五六七八九十，一二三四五六七八九十。',
                        'asr_format.wav': '事实上你会发现中国很多家庭都有些孩子真的是非常愤怒的说父母过于的掌控了。于是在这个过程当中，我学了很多心理学，也看了很多心理学的书之后才发现真正要成长的不是孩子，而是我们自己。只有我们自己成长了，我们才会懂得如何经营好家庭，如何经营好孩子。那妈妈是孩子的榜样，你自己能做到什么，孩子自然就会跟着你去学习。',
                        }

# ref_s_list = ['musk-zh1.wav']
# datas = open('wavs_new.txt','r').readlines()
# datas_list = []
# for da in tqdm.tqdm(datas):
spk = 'asr_format.wav'
spk_name = spk.replace('.wav', '')
name = f'{spk_name}_fish'
text = '中国六十岁以上的人已接近四亿，这个数字有多庞大！更惊人的是，超过百分之九十的人处于亚健康状态'
reference_text = reference_audio_dict[spk]
command = f'python3 -m tools.api_client --text "{text}" --reference_audio "{spk}"  --reference_text "{reference_text}" --streaming False --is_norm 0 '
os.system(command)
