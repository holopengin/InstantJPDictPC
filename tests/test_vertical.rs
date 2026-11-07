fn main() {
    // Quick test: NotoSansJP vertical form glyphs
    let font_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fonts/NotoSansJP-Regular.ttf");
    let font_data = std::fs::read(font_path).unwrap();
    let font = fontdue::Font::from_bytes(font_data, fontdue::FontSettings::default()).unwrap();

    let pairs = [
        ('、', '\u{FE11}'), ('。', '\u{FE12}'),
        ('「', '\u{FE43}'), ('」', '\u{FE44}'),
        ('ー', '\u{FE31}'), ('・', '\u{30FB}'),
        ('（', '\u{FE35}'), ('）', '\u{FE36}'),
        ('！', '\u{FE15}'), ('？', '\u{FE16}'),
        ('《', '\u{FE3D}'), ('》', '\u{FE3E}'),
        ('：', '\u{FE13}'), ('；', '\u{FE14}'),
    ];

    let px = 16.0;
    for &(orig, vert) in &pairs {
        let om = font.rasterize(orig, px).0;
        let vm = font.rasterize(vert, px).0;
        println!("{} U+{:04X} -> {} U+{:04X}: ({}x{}) -> ({}x{})  ok={}",
            orig, orig as u32, vert, vert as u32,
            om.width, om.height, vm.width, vm.height,
            vm.width > 0 && vm.height > 0);
    }
}
