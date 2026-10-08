//! Synchronization / framing experiment: does repeating the payload and
//! detecting on an arbitrary window recover it under trimming/mid-extract?
use audiofp_wm_bakeoff::{dsp, io};
use std::path::Path;
fn dec(src:&Path,sr:u32)->Vec<f32>{
    let t=std::env::temp_dir().join("sync.wav");
    std::process::Command::new("ffmpeg").args(["-y","-v","error","-i"]).arg(src)
        .args(["-ac","1","-ar",&sr.to_string(),"-c:a","pcm_s16le"]).arg(&t).status().unwrap();
    io::read_wav_mono_f32(&t).unwrap()
}
fn ber(a:&[u8],b:&[u8])->f64{(0..a.len()).filter(|&i|a[i]!=b[i]).count() as f64/a.len() as f64}
const NB:usize=64;
const NFFT:usize=2048;const HOP:usize=512;
fn order(key:&[u8])->Vec<usize>{
    let nb=NFFT/2+1;let lo=10;let hi=((nb as f32)*0.75) as usize;let cnt=300;
    let mut rng=dsp::ChaCha::new(&dsp::derive_key(key,b"bins"),0);
    let mut o:Vec<usize>=(lo..hi).collect();
    for i in (1..o.len()).rev(){let j=(rng.u32() as usize)%(i+1);o.swap(i,j);}
    o.truncate(cnt);o
}
fn embed(x:&[f32],payload:&[u8],key:&[u8],alpha:f32)->Vec<f32>{
    let nb=NFFT/2+1;let ord=order(key);
    let mut stft=dsp::Stft::new(NFFT,HOP);let(mut spec,nf)=stft.analyze(x);
    for f in 0..nf{let b=f%NB;let sign=if payload[b]==0{-1.0}else{1.0};
        let mut r=dsp::ChaCha::new(&dsp::derive_key(key,b"spread"),b as u64);
        for &bin in &ord{let chip=r.bipolar();let cell=&mut spec[f*nb+bin];let mag=cell.norm();
            let d=sign*chip*alpha*mag;let ph=cell.arg();*cell=rustfft::num_complex::Complex::from_polar((mag+d).max(1e-7),ph);}}
    let mut istft=dsp::Istft::new(NFFT,HOP);istft.synthesize(&spec,nf,x.len())
}
/// Detect on an arbitrary window; frames index from the window start, so the
/// bit phase is unknown. Try all `NB` bit-phase offsets and combine.
fn detect(y:&[f32],key:&[u8])->Vec<u8>{
    let nb=NFFT/2+1;let ord=order(key);
    let mut stft=dsp::Stft::new(NFFT,HOP);let(spec,nf)=stft.analyze(y);
    if nf<NB{return vec![0;NB];}
    let mut bm=vec![0.0f64;nb];
    for f in 0..nf{for &bin in &ord{bm[bin]+=spec[f*nb+bin].norm() as f64;}}
    for m in bm.iter_mut(){*m/=nf as f64;}
    // For each trial phase, accumulate per-bit correlation, then pick the
    // phase with the largest total |corr|.
    let mut best_phase=0;let mut best_energy=0.0f64;
    let mut phase_corrs=vec![[0.0f64;NB];NB];
    for ph0 in 0..NB{
        let mut corr=[0.0f64;NB];
        for f in 0..nf{
            let b=(f+ph0)%NB;
            let mut r=dsp::ChaCha::new(&dsp::derive_key(key,b"spread"),b as u64);
            for &bin in &ord{
                let chip=r.bipolar() as f64;
                let resid=spec[f*nb+bin].norm() as f64-bm[bin];
                corr[b]+=(resid/bm[bin].max(1e-9))*chip;
            }
        }
        let e: f64=corr.iter().map(|c|c.abs()).sum();
        phase_corrs[ph0]=corr;
        if e>best_energy{best_energy=e;best_phase=ph0;}
    }
    phase_corrs[best_phase].iter().map(|&c|if c>=0.0{1}else{0}).collect()
}
fn main(){
    let sr=16000;
    let mut x=dec(Path::new("assets/grieg.ogg"),sr);
    let y=dec(Path::new("assets/mozart.ogg"),sr);x.extend_from_slice(&y);
    let mut p=dsp::ChaCha::new(b"payload",0);
    let payload:Vec<u8>=(0..NB).map(|_|(p.u32()&1) as u8).collect();
    let key=b"bakeoff-secret-key-0123456789";
    let alpha=0.35;
    let wm=embed(&x,&payload,key,alpha);
    println!("full clean BER={:.3}",ber(&detect(&wm,key),&payload));
    // trims
    for &(name,delay,cut) in &[("trim-lead-0.5s",8000usize,0usize),("trim-lead-3s",48000,0),
                               ("trim-tail-3s",0,48000),("mid-10s",0,0)]{
        let start=delay;let end=if cut>0{wm.len()-cut}else{wm.len()};
        let start=if name=="mid-10s"{160000}else{start};
        let end=if name=="mid-10s"{320000}else{end};
        let seg=&wm[start..end.min(wm.len())];
        println!("{name}: len={}s BER={:.3}",seg.len()/sr as usize,ber(&detect(seg,key),&payload));
    }
    // Resample round trip via ffmpeg handled in main matrix.
}
