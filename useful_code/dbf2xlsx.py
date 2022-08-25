import struct
import datetime
import decimal
import itertools
import csv


def dbfreader(f):
    """Returns an iterator over records in a Xbase DBF file.
    The first row returned contains the field names.
    The second row contains field specs: (type, size, decimal places).
    Subsequent rows contain the data records.
    If a record is marked as deleted, it is skipped.
    File should be opened for binary reads.
    """
    # See DBF format spec at:
    #     http://www.pgts.com.au/download/public/xbase.htm#DBF_STRUCT

    numrec, lenheader = struct.unpack('<xxxxLH22x', f.read(32))
    numfields = (lenheader - 33) // 32

    fields = []
    for fieldno in xrange(numfields):
        name, typ, size, deci = struct.unpack('<11sc4xBB14x', f.read(32))
        name = name.replace('\0', '')  # eliminate NULs from string
        fields.append((name, typ, size, deci))
    yield [field[0] for field in fields]
    yield [tuple(field[1:]) for field in fields]

    terminator = f.read(1)
    assert terminator == '\r'

    fields.insert(0, ('DeletionFlag', 'C', 1, 0))
    fmt = ''.join(['%ds' % fieldinfo[2] for fieldinfo in fields])
    fmtsiz = struct.calcsize(fmt)
    for i in xrange(numrec):
        record = struct.unpack(fmt, f.read(fmtsiz))
        if record[0] != ' ':
            continue  # deleted record
        result = []
        for (name, typ, size, deci), value in itertools.izip(fields, record):
            if name == 'DeletionFlag':
                continue
            if typ == "N":
                value = value.replace('\0', '').lstrip()
                if value == '':
                    value = 0
                elif deci:
                    value = decimal.Decimal(value)
                else:
                    value = int(value)
            elif typ == 'D':
                y, m, d = int(value[:4]), int(value[4:6]), int(value[6:8])
                value = datetime.date(y, m, d)
            elif typ == 'L':
                value = (value in 'YyTt' and 'T') or (value in 'NnFf' and 'F') or '?'
            elif typ == 'F':
                value = float(value)
            result.append(value)
        yield result


def dbfwriter(f, fieldnames, fieldspecs, records):
    """ Return a string suitable for writing directly to a binary dbf file.
    File f should be open for writing in a binary mode.
    Fieldnames should be no longer than ten characters and not include \x00.
    Fieldspecs are in the form (type, size, deci) where
        type is one of:
            C for ascii character data
            M for ascii character memo data (real memo fields not supported)
            D for datetime objects
            N for ints or decimal objects
            L for logical values 'T', 'F', or '?'
        size is the field width
        deci is the number of decimal places in the provided decimal object
    Records can be an iterable over the records (sequences of field values).
    """
    # header info
    ver = 3
    now = datetime.datetime.now()
    yr, mon, day = now.year - 1900, now.month, now.day
    numrec = len(records)
    numfields = len(fieldspecs)
    lenheader = numfields * 32 + 33
    lenrecord = sum(field[1] for field in fieldspecs) + 1
    hdr = struct.pack('<BBBBLHH20x', ver, yr, mon, day, numrec, lenheader, lenrecord)
    f.write(hdr)

    # field specs
    for name, (typ, size, deci) in itertools.izip(fieldnames, fieldspecs):
        name = name.ljust(11, '\x00')
        fld = struct.pack('<11sc4xBB14x', name, typ, size, deci)
        f.write(fld)

    # terminator
    f.write('\r')

    # records
    for record in records:
        f.write(' ')  # deletion flag
        for (typ, size, deci), value in itertools.izip(fieldspecs, record):
            if typ == "N":
                value = str(value).rjust(size, ' ')
            elif typ == 'D':
                value = value.strftime('%Y%m%d')
            elif typ == 'L':
                value = str(value)[0].upper()
            else:
                value = str(value)[:size].ljust(size, ' ')
            assert len(value) == size
            f.write(value)

    # End of file
    f.write('\x1A')

    ###################################################################################3
def dbf2csv():
    filename = r'H:\tool\out\road_hz.dbf'
    f = open(filename, 'rb')
    db = list(dbfreader(f))
    f.close()
    for record in db:
        print record

    temp = []
    for j in range(len(db[0])):
        temp1 = []
        for i in range(2, len(db)):
            temp1.append(float(db[i][j]))
        temp.append(temp1)
    ##### fieldnames is first row means fieldname,fieldspecs is second row means fieldType,records is afterRows means records
    fieldnames, fieldspecs, records = db[0], db[1], db[2:]


    # Remove a field
    # del fieldnames[0]
    # del fieldspecs[0]
    # records = [rec[1:] for rec in records]

    # Create a new DBF
    filename1 =r'H:\tool\out\road_hz212.dbf'
    f1 = open(filename1, 'wb+')
    dbfwriter(f1, fieldnames, fieldspecs, records)

    # Read the data back from the new DBF
    print '-' * 50
    f1.seek(0)
    for line in dbfreader(f1):
        print line
    f1.close()

    # Convert to CSV
    print '.' * 50
    filename1 =r'H:\tool\out\road_hz.csv'
    f1 = open(filename1, 'wb+')
    csv.writer(f1).writerow(fieldnames)
    csv.writer(f1).writerows(records)
    # print f1.getvalue()
    f1.close()

if __name__ == '__main__':
    dbf2csv()