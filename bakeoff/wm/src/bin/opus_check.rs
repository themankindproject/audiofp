//! Replicate the full-matrix Opus path exactly (16 s single track, the
//! run_codec helper) to check whether the reported Opus failure is real.
use audiofp_wm_bakeoff::{algo::Candidate, dsp, io};
use std::path::{Path, PathBuf};
fn dec(src:&Path,sr:u32)->Vec<f32>{
    let t=std::env::temp_dir().join("oc.wav");
    std::process::Command::new("ffmpeg").args(["-y","-v","error","-i"]).arg(src)
        .args(["-ac","1","-ar",&sr.to_string(),"-c:a","pcm_s16le"]).arg(&t).status().unwrap();
    io::read_wav_mono_f32(&t).unwrap()
}
fn ber(a:&[u8],b:&[u8])->f64{(0..a.len()).filter(|&i|a[i]!=b[i]).count() as f64/a.len() as f64}
fn snr(a:&[f32],b:&[f32])->f64{let n=a.len().min(b.len());let mut s=0.0;let mut e=0.0;for i in 0..n{s+=(a[i] as f64).powi(2);e+=((a[i]-b[i]) as f64).powi(2);}10.0*(s/e).log10()}
fn main(){
    let sr=16000;
    let x=dec(Path::new("assets/galway.flac"),sr);
    println!("track len={}s",x.len()/sr as usize);
    let mut p=dsp::ChaCha::new(b"payload",0);
    let payload:Vec<u8>=(0..64).map(|_|(p.u32()&1) as u8).collect();
    let key=b"bakeoff-secret-key-0123456789";
    // Exact full-matrix candidate config.
    let mut c=audiofp_wm_bakeoff::algo::MagDsss::new();
    c.alpha=0.35; c.bins_per_frame=300; c.bin_hi_frac=0.75; c.bin_lo=10;
    let (wm,_)=c.embed(&x,&payload,key);
    println!("clean BER={:.3}",ber(&c.detect(&wm,64,key).0,&payload));
    let tmp=std::env::temp_dir();
    let wmw=tmp.join("oc_wm.wav");io::write_wav_mono_f32(&wmw,&wm,sr).unwrap();
    for (name,args,ext) in [
        ("opus-32",vec!["-c:a","libopus","-b:a","32k"],"opus"),
        ("opus-64",vec!["-c:a","libopus","-b:a","64k"],"opus"),
        ("opus-96",vec!["-c:a","libopus","-b:a","96k"],"opus"),
        ("opus-128",vec!["-c:a","libopus","-b:a","128k"],"opus"),
    ]{
        let enc:PathBuf=wmw.with_extension(ext);
        let st=std::process::Command::new("ffmpeg").args(["-y","-v","error","-i"]).arg(&wmw)
            .args(&args).arg(&enc).status().unwrap();
        println!("  encode {name} ok={}",st.success());
        let out=tmp.join(format!("oc_{name}.wav"));
        let st2=std::process::Command::new("ffmpeg").args(["-y","-v","error","-i"]).arg(&enc)
            .args(["-ac","1","-c:a","pcm_s16le"]).arg(&out).status().unwrap();
        println!("  decode {name} ok={}",st2.success());
        let d=io::read_wav_mono_f32(&out).unwrap();
        println!("{name}: BER={:.3} snr={:.1}dB len={}s",ber(&c.detect(&d,64,key).0,&payload),snr(&x[..x.len().min(d.len())],&d[..x.len().min(d.len())]),d.len()/sr as usize);
        // Show probe info: what sample rate did the decoded file have?
        let probe=std::process::Command::new("ffprobe").args(["-v","error","-select_streams","a:0",
            "-show_entries","stream=sample_rate,channels","-of","csv=p=0"]).arg(&enc).output().unwrap();
        println!("  encoded stream sr/ch: {}",String::from_utf8_lossy(&probe.stdout).trim());
    }
}
