# 04: Experimental data from images (FFT maps, spectra, decks)

```
The experimental rotation data are an image: <image or deck path; slide/page>.
1. If it is a deck, use `experiment_data.extract_slide_deck(<deck>, <folder>)` (or extract PDF pages) and show me slides.md; identify the
   FFT-vs-angle image and any LK tables in the slide text.
2. Calibrate the image: give the pixel positions of two frequency ticks, two angle ticks, the colour-bar
   column with its 0 and max pixels, and the data range; say how you found them (tick detection, axis lines).
3. Call `digitize_fft_map(image, ...)` at the MEASURED angles only. Validate against the LK
   tables (median |dev| should be <~1-2 %), list failures, and report possible striping runs.
4. Write the canonical exp_sweep_peaks.csv with a header stating that the values are DIGITISED, from which
   file/slide, and how accurate the frequencies and magnitudes are.
```

**Check:** look at the validation table yourself; ask for the raw FFT data if possible.
