//! Targeted follow-ups: (1) does synchronization fix temporal attacks on the
//! winning mag_dsss design? (2) why does Opus fail and can a wider/robust
//! bin band or stronger alpha fix it?
use audiofp_wm_bakeoff::{dsp, io};
use std::path::Path;
fn dec(src:&Path,sr:u32)->Vec<f32>{
    let t=std::env::temp_dir().join("gaps.wav");
    std::process::Command::new("ffmpeg").args(["-y","-v","error","-i"]).arg(src)
        .args(["-ac","1","-ar",&sr.to_string(),"-c:a","pcm_s16le"]).arg(&t).status().unwrap();
    io::read_wav_mono_f32(&t).unwrap()
}
fn ber(a:&[u8],b:&[u8])->f64{(0..a.len()).filter(|&i|a[i]!=b[i]).count() as f64/a.len() as f64}
const NB:usize=64;const NFFT:usize=2048;const HOP:usize=512;
struct Cfg{lo:usize,hi_frac:f32,cnt:usize,alpha:f32}
fn order(key:&[u8],c:&Cfg)->Vec<usize>{
    let nb=NFFT/2+1;let hi=((nb as f32)*c.hi_frac) as usize;
    let mut rng=dsp::ChaCha::new(&dsp::derive_key(key,b"bins"),0);
    let mut o:Vec<usize>=(c.lo..hi).collect();
    for i in (1..o.len()).rev(){let j=(rng.u32() as usize)%(i+1);o.swap(i,j);}
    o.truncate(c.cnt);o
}
fn embed(x:&[f32],payload:&[u8],key:&[u8],c:&Cfg)->Vec<f32>{
    let nb=NFFT/2+1;let ord=order(key,c);
    let mut stft=dsp::Stft::new(NFFT,HOP);let(mut spec,nf)=stft.analyze(x);
    for f in 0..nf{let b=f%NB;let sign=if payload[b]==0{-1.0}else{1.0};
        let mut r=dsp::ChaCha::new(&dsp::derive_key(key,b"spread"),b as u64);
        for &bin in &ord{let chip=r.bipolar();let cell=&mut spec[f*nb+bin];let mag=cell.norm();
            let d=sign*chip*c.alpha*mag;let ph=cell.arg();*cell=rustfft::num_complex::Complex::from_polar((mag+d).max(1e-7),ph);}}
    let mut istft=dsp::Istft::new(NFFT,HOP);istft.synthesize(&spec,nf,x.len())
}
fn detect(y:&[f32],key:&[u8],c:&Cfg)->Vec<u8>{
    let nb=NFFT/2+1;let ord=order(key,c);
    let mut stft=dsp::Stft::new(NFFT,HOP);let(spec,nf)=stft.analyze(y);
    if nf<NB{return vec![0;NB];}
    let mut bm=vec![0.0f64;nb];
    for f in 0..nf{for &bin in &ord{bm[bin]+=spec[f*nb+bin].norm() as f64;}}
    for m in bm.iter_mut(){*m/=nf as f64;}
    let mut best=[0.0f64;NB];let mut best_e=-1.0;
    for ph0 in 0..NB{
        let mut corr=[0.0f64;NB];
        for f in 0..nf{let b=(f+ph0)%NB;
            let mut r=dsp::ChaCha::new(&dsp::derive_key(key,b"spread"),b as u64);
            for &bin in &ord{let chip=r.bipolar() as f64;
                let resid=spec[f*nb+bin].norm() as f64-bm[bin];
                corr[b]+=(resid/bm[bin].max(1e-9))*chip;}}
        let e:f64=corr.iter().map(|x|x.abs()).sum();
        if e>best_e{best_e=e;best=corr;}
    }
    best.iter().map(|&x|if x>=0.0{1}else{0}).collect()
}
fn main(){
    let sr=16000;
    let mut x=dec(Path::new("assets/grieg.ogg"),sr);
    let y=dec(Path::new("assets/mozart.ogg"),sr);x.extend_from_slice(&y);
    let mut p=dsp::ChaCha::new(b"payload",0);
    let payload:Vec<u8>=(0..NB).map(|_|(p.u32()&1) as u8).collect();
    let key=b"bakeoff-secret-key-0123456789";
    let tmp=std::env::temp_dir();

    println!("== (1) sync + temporal attacks (alpha 0.35) ==");
    let c=Cfg{lo:10,hi_frac:0.75,cnt:300,alpha:0.35};
    let wm=embed(&x,&payload,key,&c);
    // write + attacks
    let wmw=tmp.join("g_wm.wav");io::write_wav_mono_f32(&wmw,&wm,sr).unwrap();
    println!("clean BER={:.3}",ber(&detect(&wm,key,&c),&payload));
    for (name,filt) in [("lead-0.3",Some("atrim=start=0.3")),("lead-3.0",Some("atrim=start=3.0")),
                        ("mid-10s",Some("atrim=start=10:end=20")),("tail-3",Some("atrim=end=57"))]{
        let out=tmp.join(format!("g_{name}.wav"));
        std::process::Command::new("ffmpeg").args(["-y","-v","error","-i"]).arg(&wmw)
            .args(["-af",filt.unwrap(),"-ac","1","-ar","16000","-c:a","pcm_s16le"]).arg(&out).status().unwrap();
        let seg=io::read_wav_mono_f32(&out).unwrap();
        println!("{name}: len={}s BER={:.3}",seg.len()/sr as usize,ber(&detect(&seg,key,&c),&payload));
    }

    println!("\n== (2) Opus: does alpha / band help? ==");
    for &alpha in &[0.35f32,0.6,1.0]{
      for &(lo,hf,cnt) in &[(10usize,0.75f32,300usize),(10,0.5,150),(20,0.35,80)]{
        let c=Cfg{lo,hi_frac:hf,cnt,alpha};
        let wm=embed(&x,&payload,key,&c);
        let wmw=tmp.join("g2_wm.wav");io::write_wav_mono_f32(&wmw,&wm,sr).unwrap();
        let tmpf=tmp.join("g2.opus");
        std::process::Command::new("ffmpeg").args(["-y","-v","error","-i"]).arg(&wmw)
            .args(["-c:a","libopus","-b:a","64k"]).arg(&tmpf).status().unwrap();
        let out=tmp.join("g2_dec.wav");
        std::process::Command::new("ffmpeg").args(["-y","-v","error","-i"]).arg(&tmpf)
            .args(["-ac","1","-ar","16000","-c:a","pcm_s16le"]).arg(&out).status().unwrap();
        let d=io::read_wav_mono_f32(&out).unwrap();
        println!("alpha={alpha:.1} lo={lo} hf={hf} cnt={cnt}: opus64 BER={:.3} cleanBER={:.3}",
            ber(&detect(&d,key,&c),&payload), ber(&detect(&wm,key,&c),&payload));
      }
    }
}
