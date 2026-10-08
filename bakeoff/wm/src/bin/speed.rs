//! Throughput probe: embed + detect vs realtime, on a single core.
use audiofp_wm_bakeoff::{algo::{Candidate, MagDsss}, dsp, io};
use std::path::Path;
use std::time::Instant;
fn dec(src:&Path,sr:u32)->Vec<f32>{
    let t=std::env::temp_dir().join("sp.wav");
    std::process::Command::new("ffmpeg").args(["-y","-v","error","-i"]).arg(src)
        .args(["-ac","1","-ar",&sr.to_string(),"-c:a","pcm_s16le"]).arg(&t).status().unwrap();
    io::read_wav_mono_f32(&t).unwrap()
}
fn main(){
    let sr=16000u32;
    let base=dec(Path::new("assets/grieg.ogg"),sr);
    // Build ~300 s by tiling.
    let mut x=Vec::new();
    while x.len() < 300*sr as usize { x.extend_from_slice(&base); }
    x.truncate(300*sr as usize);
    let mut p=dsp::ChaCha::new(b"payload",0);
    let payload:Vec<u8>=(0..64).map(|_|(p.u32()&1) as u8).collect();
    let key=b"bakeoff-secret-key-0123456789";
    let mut c=MagDsss::new();

    for &secs in &[5usize,30,60,300]{
        let n=secs*sr as usize;
        let seg=&x[..n];
        // warmup
        let _=c.embed(&seg[..sr as usize*2],&payload,key);
        let t=Instant::now();
        let (wm,_)=c.embed(seg,&payload,key);
        let te=t.elapsed().as_secs_f64();
        let t2=Instant::now();
        let _=c.detect(&wm,64,key);
        let td=t2.elapsed().as_secs_f64();
        println!("{secs:>4}s: embed {:.1} ms ({:8.1}x rt) | detect {:.1} ms ({:8.1}x rt)",
            te*1000.0, secs as f64/te, td*1000.0, secs as f64/td);
    }
    // Sync-on detection is the expensive path (n_bits phase trials).
    let mut c2=MagDsss::new(); c2.sync=false;
    let seg=&x[..30*sr as usize];
    let (wm,_)=c2.embed(seg,&payload,key);
    let t=Instant::now(); let _=c2.detect(&wm,64,key); let td=t.elapsed().as_secs_f64();
    println!("30s detect WITHOUT sync: {:.1} ms ({:.1}x rt)",td*1000.0,30.0/td);
}
