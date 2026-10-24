pub mod audio_gen;
#[cfg(all(
    feature = "std-wav",
    feature = "std-mp3",
    feature = "std-flac",
    feature = "std-ogg"
))]
pub mod catalog;
pub mod degrade;
pub mod temp;
