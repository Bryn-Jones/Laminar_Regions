# Laminar_Regions

Create metadata for su2 (structured) mesh points that identifies to what extent they belong to a "laminar flow region".

The mesh is divided up spanwise by the user in order to identify areas where "three-dimensional effects" cause fully turbulent flow. Betwixt these regions are "laminar flow regions". The mesh points in these regions will then be given a number from 0 to 1 indicating how close (distance) to the leading edge from the trailing edge they are, allowing for arbitrary laminar transition points to be chosen.

The code now also takes in csv files pertaining to updated node positions, field data (e.g. Cp), and 2D simulations, and can write corrected data to a VTK file based on the interpolation of the 2D simulation parameters.
