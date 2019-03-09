import numpy as np
import matplotlib.pyplot as mp
import scipy.io.wavfile as wf
import numpy.fft as nf

sample_rate,noised_sigs=wf.read('noised.wav')
print(sample_rate,noised_sigs.shape)

#声音在存储时放大了2**15,所以在处理时先除去
noised_sigs=noised_sigs/2**15

#1.绘制音频的时域图像
times=np.arange(len(noised_sigs))/sample_rate
mp.figure('Filter',facecolor='lightgray')
mp.subplot(2,2,1)
mp.title('Noised',fontsize=16)
mp.xlabel('Times',fontsize=12)
mp.ylabel('Signal',fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=":")
mp.plot(times[:178],noised_sigs[:178],color='dodgerblue',label='Signal')
mp.tight_layout()
mp.legend()

#2.基于傅里叶变换，获取音频频率信息，绘制音频频域的频率/能量图像
# 通过采样个数与周期得到频率数组
freqs=nf.fftfreq(times.size,1/sample_rate)
print(freqs.size)
#获取每个频率对应的能量值
noised_fft=nf.fft(noised_sigs)
noised_pow=np.abs(noised_fft)
mp.subplot(2,2,2)
mp.title('Frequency',fontsize=16)
mp.xlabel('Frequency',fontsize=12)
mp.ylabel('Power',fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=":")
mp.semilogy(freqs[freqs>0],noised_pow[freqs>0],color='orangered',label='Frequency')
mp.tight_layout()
mp.legend()

#3.将低频噪声去除后绘制音频频域的频率/能量图像
fund_freq=freqs[noised_pow.argmax()]
# print(fund_freq)
#找到不是高能信号的噪声信号
noised_index=np.where(freqs!=fund_freq)
#把fft得到的复数数组中所有噪声信号的值改为0
filter_fft=noised_fft.copy()
filter_fft[noised_index]=0
filter_pow=np.abs(filter_fft)
mp.subplot(2,2,4)
mp.title('Filter Frequency',fontsize=16)
mp.xlabel('Filter Frequency',fontsize=12)
mp.ylabel('Filter Power',fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=":")
mp.plot(freqs[freqs>0],filter_pow[freqs>0],color='orangered',label='Filter Frequency')
mp.tight_layout()
mp.legend()

#4.基于逆向傅里叶变换，生成新的音频信号，绘制音频的时域图(时间/位移图像)
filter_sigs=nf.ifft(filter_fft).real
mp.subplot(2,2,3)
mp.title('Filter',fontsize=16)
mp.xlabel('Times',fontsize=12)
mp.ylabel('Signal',fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=":")
mp.plot(times[:178],filter_sigs[:178],color='dodgerblue',label='Signal')
mp.tight_layout()
mp.legend(loc='upper right')

# 5.生成音频文件
wf.write('filter.wav',sample_rate,(filter_sigs*2**25).astype('int16'))

mp.show()
