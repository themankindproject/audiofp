use audiofp_wm_bakeoff::{dsp, io};
use std::path::Path;
fn dec(src:&Path,sr:u32)->Vec<f32>{
    let t=std::env::temp_dir().join("q.wav");
    std::process::Command::new("ffmpeg").args(["-y","-v","error","-i"]).arg(src)
        .args(["-ac","1","-ar",&sr.to_string(),"-c:a","pcm_s16le"]).arg(&t).status().unwrap();
    io::read_wav_mono_f32(&t).unwrap()
}
fn snr(a:&[f32],b:&[f32])->f64{let n=a.len().min(b.len());let mut s=0.0;let mut e=0.0;for i in 0..n{s+=(a[i] as f64).powi(2);e+=((a[i]-b[i]) as f64).powi(2);}10.0*(s/e).log10()}
fn main(){
    let sr=16000;
    let x=dec(Path::new("assets/galway.flac"),sr);
    let y=dec(Path::new("assets/grieg.ogg"),sr);
    let key=b"k";
    let payload:Vec<u8>=(0..64).map(|i|(i%3==0) as u8).collect();
    for &(name,track) in &[("galway",&x),("grieg",&y)]{
        for &alpha in &[0.1f32,0.2,0.4,0.8]{
            let n_fft=2048;let hop=512;let nb=n_fft/2+1;
            let bin_lo=10;let bin_hi=((nb as f32)*0.75) as usize;let bpf=300;
            let mut stft=dsp::Stft::new(n_fft,hop);let (mut spec,nf)=stft.analyze(track);
            let mut rng=dsp::ChaCha::new(&dsp::derive_key(key,b"bins"),0);
            let mut order:Vec<usize>=(bin_lo..bin_hi).collect();
            for i in (1..order.len()).rev(){let j=(rng.u32() as usize)%(i+1);order.swap(i,j);}
            order.truncate(bpf);
            for f in 0..nf{let b=f%64;let sign=if payload[b]==0{-1.0}else{1.0};
                let mut r2=dsp::ChaCha::new(&dsp::derive_key(key,b"spread"),b as u64);
                for &bin in order.iter(){let chip=r2.bipolar();let cell=&mut spec[f*nb+bin];let mag=cell.norm();
                    let delta=sign*chip*alpha*mag;let ph=cell.arg();*cell=rustfft::num_complex::Complex::from_polar((mag+delta).max(1e-7),ph);}}
            let mut istft=dsp::Istft::new(n_fft,hop);let wm=istft.synthesize(&spec,nf,track.len());
            let lsd=audiofp_wm_bakeoff::attack::log_spectral_distance(track,&wm,1024,256);
            // 16-bit requantisation distortion floor
            let q:Vec<f32>=wm.iter().map(|&s|(s.clamp(-1.0,1.0)*32767.0).round()/32767.0).collect();
            println!("{name} alpha={alpha:.1} SNR={:6.1}dB LSD={lsd:.2}dB requant_snr={:6.1}dB",snr(track,&wm),snr(track,&q));
        }
    }
}
