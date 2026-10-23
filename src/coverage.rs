//! Shared file mapping survives native crashes. Exact IDs avoid map hashes.
use std::{collections::BTreeSet, fs::File, io, path::Path};

pub const MAP_SIZE: u64 = 16 * 1024 * 1024;
pub fn prepare(directory: &Path) -> io::Result<()> {
    File::create(directory.join("coverage.map"))?.set_len(MAP_SIZE + 16)
}

pub struct Coverage {
    pub guards: u32,
    pub edges: BTreeSet<String>,
}

pub fn read(directory: &Path) -> io::Result<Coverage> {
    let bytes = std::fs::read(directory.join("coverage.map"))?;
    if bytes.len() != MAP_SIZE as usize + 16 {
        return Err(io::Error::other("invalid compiler coverage map size"));
    }
    if &bytes[..8] != b"MRACOV01" {
        return Ok(Coverage {
            guards: 0,
            edges: BTreeSet::new(),
        });
    }
    let guards = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if bytes[12] != 0 || guards as u64 > MAP_SIZE {
        return Err(io::Error::other(
            "compiler coverage exceeds the 16 MiB guard map",
        ));
    }
    let edges = bytes[16..16 + guards as usize]
        .iter()
        .enumerate()
        .filter(|(_, count)| **count != 0)
        .map(|(id, _)| format!("compiler-edge:{id}"))
        .collect();
    Ok(Coverage { guards, edges })
}
