#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import click
import numpy as np
from tqdm import trange
from astropy import coordinates as coord, units as u
from astropy.io import fits
from astropy.time import Time, TimeDelta, TimezoneInfo

__author__ = 'bernardowb'
__version__ = '1.2'

def obs_info(lat_str, lon_str, elev, utc_offset):
    """
    Create 'astropy.coord.EarthLocation' and 'astropy.time.TimezoneInfo' 
    objects from observatoty geographical coordinates and timezone offset.

    Parameters
    ----------
    lat_str : str
        Observatory latitude.
    lon_str : str
        Observatory longitude.
    elev : float
        Observatory elevation (in meters).
    utc_offset : float
        Observatoty timezone offset (in hours).

    Returns
    -------
    tp_info : tuple
        Tuple with 'astropy.coord.EarthLocation' (first element) and 
        'astropy.time.TimezoneInfo' objetcs (second element).

    """

    # get observatory location and timezone
    la = coord.Latitude(lat_str, unit=u.deg)
    lo = coord.Longitude(lon_str, unit=u.deg)
    el = u.Quantity(elev, unit=u.m)
    uo = u.Quantity(utc_offset, unit=u.hour) # utc offset (hours)

    # create earthlocation and timezoneinfo objects
    oloc = coord.EarthLocation(lat=la, lon=lo, height=el)
    otmz = TimezoneInfo(utc_offset=uo)
    
    tp_info = (oloc, otmz)
    return tp_info

def nhvals(tobs, fk5f, oloc, otmz):
    """
    Evaluate new header values (formatted strings) from the date/time of 
    observation ('astropy.time.Time' object), target coordinates 
    ('astropy.coord.SkyCood' object), observatory location 
    ('astropy.coord.EarthLocation' object) and timezone offset 
    ('astropy.time.TimezoneInfo' object).

    Parameters
    ----------
    tobs : astropy.time.Time
        Date/Time of observation.
    fk5f : astropy.coord.SkyCood
        Target coordinates.
    oloc : astropy.coord.EarthLocation
        Observatory location.
    otmz : astropy.time.TimezoneInfo
        Observatoty timezone offset.

    Returns
    -------
    tp_str : tuple of strings
        Tuple of strings with UTC time (ISOT format), julian date, sidereal 
        time, hour angle, airmass and local time of observation.
    """

    # create altaz and fk5 precessed frames and coords
    altaz_frame = coord.AltAz(obstime=tobs, location=oloc)
    altaz_c = fk5f.transform_to(altaz_frame)
    fk5p_frame = coord.FK5(equinox=tobs)
    fk5p_c = fk5f.transform_to(fk5p_frame)

    # evaluate sidereal time, hour angle and local time
    st = tobs.sidereal_time('apparent') 
    ha = st - fk5p_c.ra.to(u.hourangle) 
    lt = tobs.to_datetime(timezone=otmz)

    # create output tuple of strings
    tobs_str = tobs.isot
    jd_str = f'{tobs.jd:.5f}'
    st_str = st.to_string(unit=u.hour, sep=':', precision=3)
    ha_str = ha.to_string(unit=u.hour, sep=':', precision=3)
    airmass_str = f'{altaz_c.secz:.5f}'
    lt_str = lt.strftime('%H:%M:%S')
    tp_str = (tobs_str, jd_str, st_str, ha_str, airmass_str, lt_str)

    return tp_str

def ixonslicer(fname, cardtkey='DATE-OBS', after2015=True, hextfile=None):
    """
    Slice FITS image obtained with CCD iXon in kinetic mode.

    Parameters
    ----------
    fname : str
        FITS file (full path) to be sliced.
    cardtkey : str, optional
        Keyword of the time of observation card. Default: 'DATE-OBS'
    after2015 : bool, optional
        Flag if the FITS file was observed after 2015 (i.e., the header 
        is complete). Default: True
    hextfile : str or None, optional
        ASCII file (full path) with new cards (keyword and values), to be 
        included in the headers of the output files. Default: None
    """

    finbase = os.path.basename(os.path.abspath(fname))
    finrad = os.path.splitext(finbase)[0]
    findir = os.path.dirname(os.path.abspath(fname)) + '/'
    listfilesout = []
    
    click.secho(f'\nOPD/LNA iXon FITS slicer (kinetic mode)', bold=True)
    click.secho(f'{("-" * 72)}', bold=True)
    click.echo(click.style(f'FITS file: ', bold=True) +
               click.style(f'{finbase}', bold=False))
    click.echo(click.style(f'Card time key: ', bold=True) +
               click.style(f'{cardtkey}', bold=False))
    after2015_str = click.style(f'After 2015? ', bold=True)
    if after2015:
        after2015_str += click.style(f'YES', bold=False)
    else:
        after2015_str += click.style(f'NO (header incomplete)', bold=False)
    click.echo(after2015_str)

    # open fname
    with fits.open(fname) as hdul_in:
        click.secho(f'Reading FITS file header...', fg='red', bold=False)
        hdr = hdul_in[0].header
        click.secho(f'Reading FITS file data...', fg='red', bold=False)
        data = hdul_in[0].data

        try:
            if data.ndim == 3:

                n_ima = data.shape[0]
                opd_info = obs_info('-22:32:04', '-45:34:57', 1864, -3)
                click.echo(click.style(f'Number of images: ', bold=True) +
                           click.style(f'{n_ima:d}', bold=False))
                click.echo(click.style(f'Sliced images directory: ',
                           bold=True) +
                           click.style(f'{findir}', 
                           bold=False))
           
                # get header fields values
                h_t0 = Time(hdr[cardtkey], format='isot', scale='utc',
                            location=opd_info[0])
                h_kct = TimeDelta(hdr['KCT'], format='sec')
           
                if after2015: # new ixon fits file (after 2015)
                    hdr_test = [hdr['RA'], hdr['DEC'], hdr['EPOCH']]
                    hdr_bool = [bool(not ss or ss.isspace()) 
                                for ss in hdr_test]

                    if np.any(hdr_bool):
                        click.secho(f'No values in RA, DEC or EPOCH cards.', 
                                    bold=False)   

                    else:
                        click.secho(f'Valid RA, DEC and EPOCH cards.',
                                    bold=False)
                        h_ra = coord.Angle(hdr['RA'], unit=u.hourangle)
                        h_dec = coord.Angle(hdr['DEC'], unit=u.deg)
                        h_eqx = Time(float(hdr['EPOCH']), format='jyear')

                        # create fk5 coord object
                        fk5_frame = coord.FK5(equinox=h_eqx)
                        fk5_c = coord.SkyCoord(ra=h_ra, dec=h_dec, 
                                               frame=fk5_frame)

                else: # old ixon fits file (before 2015)

                    # check if an ascii file with new header info is 
                    # being used    
                    if hextfile:

                        hdrext = fits.Header.fromtextfile(hextfile)
                        hextfbase = os.path.basename(os.path.abspath(
                                                     hextfile))
                        click.echo(click.style(f'Header file: ', bold=True) +
                                   click.style(f'{hextfbase}', bold=False))
                        click.secho(f'Cards included (keywords):', bold=True)

                        for hkwd in list(hdrext.keys()):
                            hdr[hkwd] = (hdrext[hkwd], hdrext.comments[hkwd])
                            print(f'{hkwd}')

                        h_ra = coord.Angle(hdr['RA'], unit=u.hourangle)
                        h_dec = coord.Angle(hdr['DEC'], unit=u.deg)
                        h_eqx = Time(float(hdr['EPOCH']), format='jyear')

                        # create fk5 coord object
                        fk5_frame = coord.FK5(equinox=h_eqx)
                        fk5_c = coord.SkyCoord(ra=h_ra, dec=h_dec, 
                                               frame=fk5_frame)
                    else:
                        click.echo(click.style(f'Header file: ', bold=True) +
                                   click.style(f'None', bold=False))

                # configure tqdm progress bar
                bfmt = '{l_bar}{bar}| {n_fmt}/{total_fmt}'
                bdes = f'Slicing'
                for i in trange(n_ima, ncols=70, desc=bdes, bar_format=bfmt):
                
                    # calculate respective image date-obs
                    time_obs = h_t0 + i * h_kct

                    # check if the header is complete or an ascii file 
                    # with new header info is being used
                    if (after2015 and (not np.any(hdr_bool)) or 
                        (not after2015) and hextfile):
                       # update header fields (if the header is complete or 
                       # an ASCII file is being used)
                        nhv = nhvals(time_obs, fk5_c, *opd_info)
                        hdr['DATE-OBS'] = nhv[0]

                        if (not after2015) and hextfile:
                            # create EXPTIME card if it does not exist
                            hdr.set('EXPTIME', f'{hdr["EXPOSURE"]:.5f}', 
                                    'Total Exposure Time', after='DATE-OBS')
                        
                        hdr['JD'] = nhv[1]
                        hdr['ST'] = nhv[2]
                        hdr['HA'] = nhv[3]
                        hdr['AIRMASS'] = nhv[4]

                    else:
                        # update header field
                        hdr['DATE-OBS'] = time_obs.isot
                        
                        if (not after2015) and (not hextfile):
                            # create EXPTIME card if it does not exist
                            hdr.set('EXPTIME', f'{hdr["EXPOSURE"]:.5f}',
                                    'Total Exposure Time', after='DATE-OBS')

                    # select output FITS data
                    data_out = data[i]

                    # save updated header + selected data
                    hdu_out = fits.PrimaryHDU(data_out, header=hdr)
                    hdul_out = fits.HDUList([hdu_out])
                    fileoutdir = f'{findir}{finrad}_{i:04d}.fits'
                    hdul_out.writeto(fileoutdir)
                    fileout = f'{finrad}_{i:04d}.fits'
                    listfilesout.append(fileout)
                    hdul_out.close()

            else:
                raise ValueError(f'The FITS data has' 
                                 f'{data.ndim:d} dimensions.')
    
        except ValueError as ve:
            print(ve)
    
        except OSError as oe:
            print(oe)

        else:
            click.secho(f'Sucess!', fg='green', bold=True)
            listout = f'{findir}list{finrad.upper()}'
            np.savetxt(listout, listfilesout, fmt='%s')
            click.echo(click.style(f'Images names list: ',
                       bold=True) +
                       click.style(f'list{finrad.upper()}', 
                       bold=False))
            click.secho(f'{("-" * 72)}\n', bold=True)
