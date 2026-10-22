// Plasma runs this once for each user and records it in plasmashellrc [Updates].
desktops().forEach(function (d) {
    d.wallpaperPlugin = "org.kde.slideshow";
    d.currentConfigGroup = ["Wallpaper", "org.kde.slideshow", "General"];
    d.writeConfig("SlidePaths", "/usr/share/wallpapers/DiamonDinoia/");
    d.writeConfig("SlideInterval", 300);
});
ConfigFile("kscreenlockerrc", "Greeter").writeEntry("WallpaperPlugin", "io.github.diamondinoia.spread");
