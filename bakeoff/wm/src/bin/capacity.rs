use audiofp_wm_bakeoff::{dsp};
use std::path::Path;
fn decode(src:&Path, sr:u32)->Vec<f32>{
    let tmp=std::env::temp_dir().join("cap_src.wav");
    std::process::Command::new("ffmpeg").args(["-y","-v","error","-i"]).arg(src)
        .args(["-ac","1","-ar",&sr.to_string(),"-c:a","pcm_s16le"]).arg(&tmp).status().unwrap();
    audiofp_wm_bakeoff::io::read_wav_mono_f32(&tmp).unwrap()
}
fn main(){
    let sr=16000u32;
    for f in ["assets/galway.flac","assets/grieg.ogg","assets/mozart.ogg"] {
        let x=decode(Path::new(f),sr);
        let secs=x.len() as f32/sr as f32;
        // frames at hop 512 = 31.25 fps
        let frames=(x.len()/512) as f32;
        println!("{f}: {:.1}s, ~{:.0} frames @31.25fps", secs, frames);
    }
}
