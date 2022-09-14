# Laminar_Regions

Three-dimensional laminar corrections module for the RHEA project.

"Lamiar regions" are defined in terms of upper/lower surface chordwise transition points, and spanwise "turbulence regions" caused by three-dimensional effects.

A structured su2 mesh is combined with an a-star search method to identify the nodes belonging to certain groups: leading edge, upper trailing edge, lower trailing edge, upper surface, and lower surface.

A "snapshot" su2 result file is read, whose node IDs correspond to the same ones on the su2 mesh (but not necessarily the same node coordinates). Then, every node in the upper/lower surface groups is given a correction if inside a laminar region, based on a correction factor provided by the ratio of two two-dimensional analyses, one free transition, and the other fully turbulent. Cubic splines (scipy) are used to interpolate the edges and the two-dimensional data.

Additionally, the edges of the corrected regions are smoothed via sigmoid functions with fixed end points based on a linear interpolation of the full surface solution at either end. The linear interpolator uses a tree data structure to find the nearest nodes to a given point in space. Any nodes not identified as belonging to a surface, or not inside a laminar region or smoothing region, will simply have the snapshot data copied directly over.
