# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 09:49:45 2023

@author: Marie Roch

"""

"""
AudioStreamDescriptor

Classes for determining information about audio files.
These classes can analyze various audio file formats and determine
information that is needed for constructing Streams objects.

In general, for a group of files, construct a audio header object of the
appropriate type and call method get_stream_descriptor which returns
information that can be passed to the add_file method of a Streams object.
Construct a SampleStream object with the Stream object once all files have been
added
"""
import struct,math
from datetime import datetime,timedelta
import re

# Add on imports
import numpy as np
import soundfile

class XWAVhdr:
    """
    XWAVhdr - Class for reading the extended wav format developed by
    Sean Wiggins of the Scripps Whale Acoustics Lab: www.cetus.edu
    The XWAV format places an additional chunk in the wav header and
    enables extra metadata while not preventing most audio readers from
    handling the data correctly as the chunk is ignored.
    """

    def __init__(self,filename):
        self.filename = filename
        self.xhd = {}
        self.raw = {}
        with open(filename,'rb') as f:
            self.xhd["ChunkID"] = struct.unpack("<4s",f.read(4))[0].decode("utf-8") 
            self.xhd["ChunkSize"] = struct.unpack("<I",f.read(4))[0]
            self.xhd["Format"] = struct.unpack("<4s",f.read(4))[0].decode("utf-8") 
            self.xhd["fSubchunkID"] = struct.unpack("<4s",f.read(4))[0].decode("utf-8") 
            self.xhd["fSubchunkSize"] = struct.unpack("<I",f.read(4))[0]
            self.xhd["AudioFormat"] = struct.unpack("<H",f.read(2))[0]
            self.xhd["NumChannels"] = struct.unpack("<H",f.read(2))[0]
            self.xhd["SampleRate"] = struct.unpack("<I",f.read(4))[0]
            self.xhd["ByteRate"] = struct.unpack("<I",f.read(4))[0]
            self.xhd["BlockAlign"] = struct.unpack("<H",f.read(2))[0]
            self.xhd["BitsPerSample"] = struct.unpack("<H",f.read(2))[0]
            self.nBits = self.xhd["BitsPerSample"]
            self.samp = {}
            self.samp["byte"] = math.floor(self.nBits/8)
            self.xhd["hSubchunkID"] = struct.unpack("<4s",f.read(4))[0].decode("utf-8") 
            self.xhd["hSubchunkSize"] = struct.unpack("<I",f.read(4))[0]
            self.xhd["WavVersionNumber"] = struct.unpack("<B",f.read(1))[0]
            self.xhd["FirmwareVersionNumber"] = struct.unpack("<10s",f.read(10))[0].decode("utf-8") 
            self.xhd["InstrumentID"] = struct.unpack("<4s",f.read(4))[0].decode("utf-8")
            self.xhd["SiteName"] = struct.unpack("<4s",f.read(4))[0].decode("utf-8") 
            self.xhd["ExperimentName"] = struct.unpack("<8s",f.read(8))[0].decode("utf-8") 
            self.xhd["DiskSequenceNumber"] = struct.unpack("<B",f.read(1))[0]
            self.xhd["DiskSerialNumber"] = struct.unpack("<8s",f.read(8))[0]
            self.xhd["NumOfRawFiles"] = struct.unpack("<H",f.read(2))[0]
            self.xhd["Longitude"] = struct.unpack("<i",f.read(4))[0]
            self.xhd["Latitude"] = struct.unpack("<i",f.read(4))[0]
            self.xhd["Depth"] = struct.unpack("<h",f.read(2))[0]
            self.xhd["Reserved"] = struct.unpack("<8s",f.read(8))[0]
            #Setup raw file information
            self.xhd["year"] = []
            self.xhd["month"] = []
            self.xhd["day"] = []
            self.xhd["hour"] = []
            self.xhd["minute"] = []
            self.xhd["secs"] = []
            self.xhd["ticks"] = []
            self.xhd["byte_loc"] = []
            self.xhd["byte_length"] = []
            self.xhd["write_length"] = []
            self.xhd["sample_rate"] = []
            self.xhd["gain"] = []
            self.xhd["padding"] = []
            self.raw["dnumStart"] = []
            #self.raw["dvecStart"] = []
            self.raw["dnumEnd"] = []
            #self.raw["dvecEnd"] = []
            for i in range(0,self.xhd["NumOfRawFiles"]):
                self.xhd["year"].append(struct.unpack("<B",f.read(1))[0])
                self.xhd["month"].append(struct.unpack("<B",f.read(1))[0])
                self.xhd["day"].append(struct.unpack("<B",f.read(1))[0])
                self.xhd["hour"].append(struct.unpack("<B",f.read(1))[0])
                self.xhd["minute"].append(struct.unpack("<B",f.read(1))[0])
                self.xhd["secs"].append(struct.unpack("<B",f.read(1))[0])
                self.xhd["ticks"].append(struct.unpack("<H",f.read(2))[0])
                self.xhd["byte_loc"].append(struct.unpack("<I",f.read(4))[0])
                self.xhd["byte_length"].append(struct.unpack("<I",f.read(4))[0])
                self.xhd["write_length"].append(struct.unpack("<I",f.read(4))[0])
                self.xhd["sample_rate"].append(struct.unpack("<I",f.read(4))[0])
                self.xhd["gain"].append(struct.unpack("<B",f.read(1))[0])
                self.xhd["padding"].append(struct.unpack("<7s",f.read(7))[0])
            
                self.raw["dnumStart"].append(datetime(self.xhd["year"][i]+2000,self.xhd["month"][i],self.xhd["day"][i],self.xhd["hour"][i],self.xhd["minute"][i],self.xhd["secs"][i],self.xhd["ticks"][i]*1000))
                self.raw["dnumEnd"].append(self.raw["dnumStart"][i]  + timedelta(seconds=((self.xhd["byte_length"][i]-2)/self.xhd["ByteRate"])))
            
            self.xhd["dSubchunkID"] = struct.unpack("<4s",f.read(4))[0].decode("utf-8")
            self.xhd["dSubchunkSize"] = struct.unpack("<I",f.read(4))[0]
            self.dtimeStart = self.raw["dnumStart"][0]
            self.dtimeEnd = self.raw["dnumEnd"][-1]
            
    def get_stream_descriptor(self):
        """get_stream_descriptor()
        Build a stream description suitable for passing to the add_file
        method of Streams

        :return: (filename, samples, timestamps, Fs)
        filename - Naem of XWAV
        samples - Number of samples in each contiguous segment
        timestamps - Start time of each contiguous segment
        Fs - sample rate
        """
        samples = []
        timestamps = []

        Fs = self.xhd['SampleRate']
        delta_sample = timedelta(seconds = 1.0/Fs)

        bits_per_byte = 8
        bytes_per_sample = self.xhd['NumChannels'] * \
                           self.xhd['BitsPerSample'] / bits_per_byte
        # Compute the number of samples in each raw fil
        raw_samples = [int(b / bytes_per_sample) for b in self.xhd['byte_length']]

        # This was a test case used in debugging.  We leave it here
        # (commented) in case we ever need to reverify this code:
        #
        # Five blocks:
        # 10:00-10:20, 10:20-10:40, 11:00-11:30, 11:30-12:00, 13:00-14:00
        # with 100 samples every 20 m
        #
        # It should produce three contigs:
        # starts
        #  [datetime.datetime(1, 1, 1, 10, 0),
        #   datetime.datetime(1, 1, 1, 11, 0),
        #   datetime.datetime(1, 1, 1, 13, 0)]
        # sample_blocks
        #   [200, 300, 300]
        #
        # Data structures to support this:
        # self.raw['dnumStart'] = [datetime(1,1,1,10), datetime(1,1,1,10,20),
        #                         datetime(1,1,1,11), datetime(1,1,1,11,30),
        #                         datetime(1,1,1,13)]
        # self.raw['dnumEnd'] = [datetime(1,1,1,10,20), datetime(1,1,1,10,40),
        #                      datetime(1,1,1,11,30), datetime(1,1,1,12),
        #                      datetime(1,1,1,14)]
        # raw_samples = [100,100,150,150,300]

        cum_raw_samples = np.cumsum(raw_samples)

        # Build up star times for each raw file
        # Find times with gaps:

        raw_gaps = [e - s > delta_sample
                    for s, e in zip(self.raw['dnumEnd'][0:-1],
                                    self.raw['dnumStart'][1:])]
        # gaps_at:  indices of the last block of data before a gap
        # gaps_at + 1 : indices of next contiguous block of data
        gaps_at = [idx for idx, val in
                   enumerate(raw_gaps) if val == True]
        if gaps_at is None:
            sample_blocks = [cum_raw_samples[-1]]
            starts = [self.dtimeStart]
        else:
            # There are gaps in the time series
            start_idx = 0  # idx first block of contiguous data
            sample_blocks = []  # lengths of contigs
            starts = []  # start times of each contig
            last_cum = 0  # cumulative samples in previous contig
            for gap_idx in gaps_at:
                # Add start time of contig before gap
                starts.append(self.raw['dnumStart'][start_idx])
                # Number of cumulative samples across all contigs up to
                # and including this current contig whose start time
                # we just appended.
                new_cum = cum_raw_samples[gap_idx]
                # Subtract off prior contigs to get length of this one
                block_len = new_cum - last_cum
                sample_blocks.append(block_len)  # samples in this contig
                # prepare to process next contig which starts after gap_idx
                start_idx = gap_idx + 1
                last_cum = new_cum

            # Handle last block
            starts.append(self.raw['dnumStart'][start_idx])
            sample_blocks.append(cum_raw_samples[-1] - last_cum)

        return (self.filename, sample_blocks, starts, Fs)


class WAVhdr:
    # Default timestamp parser
    default_re = re.compile(""".*
        (?P<year>\d\d(\d\d)?)[-_]?(?P<month>\d\d)[-_]?(?P<day>\d\d)
        .(?P<hour>\d\d)[:]?(?P<min>\d\d)[:]?(?P<sec>\d\d)
        \..*
        """, re.RegexFlag.VERBOSE)

    def __init__(self, filename, timestamp=None, tstamp_re=None):
        """
        WAVhdr - Read a waveform header. Will try to infer the start time
        of the data if date is not provided
        :param filename:  path to wav file
        :param timestamp: datetime of start
        :param tstamp_re: regular expression object for extracting date from
           filename.  Must support the following named patterns:
           year, month, day, hour, min, sec.

        If neither timestamp nor tstamp_re is given, a default date pattern
        will be matched.  The default pattern expects timestamps containing
        a timestamp along the following lines:
        2020-08-11 12:34:56
        20200811 123456
        200811-12:34:56
        All of these will be parsed as August 11 2020 at 12:34:56.
        If the default regular expression does not meet you needs, you can
        provide your own.  You may wish to see this class's default_re, read
        about Python re if you are not familiar with it, and try things on
        the web site regex101.com.
        """
        self.filename = filename
        self.info = soundfile.info(filename)

        if timestamp is not None:
            self.start = timestamp
        else:
            if tstamp_re is None:
                tstamp_re = self.default_re
            m = tstamp_re.match(filename)
            if m is None:
                raise ValueError("Could not derive timestamp from file")
            else:
                year = int(m.group('year'))
                if year < 100:
                    # Get current century
                    century = int(datetime.now().year / 100) * 100
                    year = century + year
                self.start = datetime(
                    year, int(m.group("month")), int(m.group("day")),
                    int(m.group("hour")), int(m.group("min")),
                    int(m.group("sec")), 0
                )

    def get_stream_descriptor(self):
        return (self.filename, self.info.frames, self.start, self.info.samplerate)