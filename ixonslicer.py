#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from tqdm import trange
from astropy import coordinates as coord, units as u
from astropy.io import fits
from astropy.time import Time, TimeDelta, TimezoneInfo

__author__ = 'bernardowb'
__version__ = '1.0'

def obs_info(lat_str, lon_str, elev, utc_offset):
    
    # gets observatory location and timezone
    la = coord.Latitude(lat_str, unit=u.deg)
    lo = coord.Longitude(lon_str, unit=u.deg)
    el = u.Quantity(elev, unit=u.m)
    uo = u.Quantity(utc_offset, unit=u.hour) # utc offset (hours)

    # creates earthlocation and timezoneinfo objects
    oloc = coord.EarthLocation(lat=la, lon=lo, height=el)
    otmz = TimezoneInfo(utc_offset=uo)
    return (oloc, otmz)

def nhvals(tobs, fk5f, oloc, otmz):

    # creates altaz and fk5 precessed frames and coords
    altaz_frame = coord.AltAz(obstime=tobs, location=oloc)
    altaz_c = fk5f.transform_to(altaz_frame)
    fk5p_frame = coord.FK5(equinox=tobs)
    fk5p_c = fk5f.transform_to(fk5p_frame)

    # evaluates sidereal time, hour angle and local time
    st = tobs.sidereal_time('apparent') 
    ha = st - fk5p_c.ra.to(u.hourangle) 
    lt = tobs.to_datetime(timezone=otmz)

    # creates output tuple of strings
    tobs_str = tobs.isot
    jd_str = f'{tobs.jd:.5f}'
    st_str = st.to_string(unit=u.hour, sep=':', precision=3)
    ha_str = ha.to_string(unit=u.hour, sep=':', precision=3)
    airmass_str = f'{altaz_c.secz:.5f}'
    lt_str = lt.strftime('%H:%M:%S')
    tp_str = (tobs_str, jd_str, st_str, ha_str, airmass_str, lt_str)

    return tp_str

def ixonslicer(fname):
    
    finbase = os.path.basename(os.path.abspath(fname))
    finrad = os.path.splitext(finbase)[0]
    findir = os.path.dirname(os.path.abspath(fname)) + '/'

    # opens fname
    with fits.open(fname) as hdul_in:
        hdr = hdul_in[0].header
        data = hdul_in[0].data

        try:
            if data.ndim == 3:

                n_ima = data.shape[0]
                opd_info = obs_info('-22:32:04', '-45:34:57', 1864, -3)

                print(f'\nOPD/LNA iXon FITS slicer (kinetic mode)\n'
                      f'{("-" * 72)}\n'
                      f'File: {finbase} ({n_ima:d} images)')

                # gets header fields values
                h_t0 = Time(hdr['DATE-OBS'], format='isot', scale='utc',
                            location=opd_info[0])
                h_kct = TimeDelta(hdr['KCT'], format='sec')
                h_ra = coord.Angle(hdr['RA'], unit=u.hourangle)
                h_dec = coord.Angle(hdr['DEC'], unit=u.deg)
                h_eqx = Time(float(hdr['EPOCH']), format='jyear')

                # creates fk5 coord object
                fk5_frame = coord.FK5(equinox=h_eqx)
                fk5_c = coord.SkyCoord(ra=h_ra, dec=h_dec, frame=fk5_frame)

                # configures tqdm progress bar
                bfmt = '{l_bar}{bar}| {n_fmt}/{total_fmt}'
                bdes = f'Slicing'
                for i in trange(n_ima, ncols=70, desc=bdes, bar_format=bfmt):
                
                    # calculates respective image date-obs
                    time_obs = h_t0 + i * h_kct

                    # updates header fields
                    nhv = nhvals(time_obs, fk5_c, *opd_info)
                    hdr['DATE-OBS'] = nhv[0]
                    hdr['JD'] = nhv[1]
                    hdr['ST'] = nhv[2]
                    hdr['HA'] = nhv[3]
                    hdr['AIRMASS'] = nhv[4]

                    # selects output FITS data
                    data_out = data[i]

                    # saves updated header + selected data
                    hdu_out = fits.PrimaryHDU(data_out, header=hdr)
                    hdul_out = fits.HDUList([hdu_out])
                    fileout = f'{findir}{finrad}_{i:04d}.fits'
                    hdul_out.writeto(fileout)
                    hdul_out.close()

            else:
                raise ValueError(f'The FITS data has' 
                                 f'{data.ndim:d} dimensions.')
    
        except ValueError as ve:
            print(ve)
    
        except OSError as oe:
            print(oe)

        else:
            print(f'Sucess!\n{("-" * 72)}\n')
