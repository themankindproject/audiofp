//! Measure the true 16-bit PCM quantisation floor, to compare against the
//! watermark distortion rather than assuming they coincide.
use audiofp_wm_bakeoff::{algo::{Candidate, MagDsss}, dsp, io};
use std::path::Path;
fn dec(src:&Path,sr:u32)->Vec<f32>{
    let t=std::env::temp_dir().join("floor.wav");
    std::process::Command::new("ffmpeg").args(["-y","-v","error","-i"]).arg(src)
        .args(["-ac","1","-ar",&sr.to_string(),"-c:a","pcm_s16le"]).arg(&t).status().unwrap();
    io::read_wav_mono_f32(&t).unwrap()
}
fn snr(a:&[f32],b:&[f32])->f64{let n=a.len().min(b.len());let mut s=0.0;let mut e=0.0;for i in 0..n{s+=(a[i] as f64).powi(2);e+=((a[i]-b[i]) as f64).powi(2);}10.0*(s/e).log10()}
fn main(){
    let sr=16000;let key=b"bakeoff-secret-key-0123456789";
    let mut p=dsp::ChaCha::new(b"payload",0);
    let payload:Vec<u8>=(0..64).map(|_|(p.u32()&1) as u8).collect();
    let mut c=MagDsss::new();
    for t in ["assets/galway.flac","assets/grieg.ogg"]{
        let x=dec(Path::new(t),sr);
        let rms=(x.iter().map(|v|(*v as f64).powi(2)).sum::<f64>()/x.len() as f64).sqrt();
        // True 16-bit floor: requantise the ORIGINAL, measure error vs original.
        let xq:Vec<f32>=x.iter().map(|&s|(s.clamp(-1.0,1.0)*32767.0).round()/32767.0).collect();
        println!("{t}: rms={rms:.4}  16-bit floor SNR = {:.1} dB", snr(&x,&xq));
        for &alpha in &[0.1f32,0.2,0.35,0.8]{
            c.alpha=alpha;
            let (wm,_)=c.embed(&x,&payload,key);
            println!("   alpha={alpha:.2}: watermark SNR = {:.1} dB", snr(&x,&wm));
        }
    }
}
