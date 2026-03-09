export_2d = false; 
L = 10.0;
W = 2.0;

module PhononicCrystal() {
    cube([L, W, 1]);
}

if (export_2d) {
    projection(cut=true) translate([0, 0, -0.5]) 
        PhononicCrystal($fn=30);
} else {
    PhononicCrystal($fn=100);
}
