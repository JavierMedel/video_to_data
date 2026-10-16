import unittest
import numpy as np
from track1_package_from_parent import validate_pair, validate_packed_rows


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.p={'pose':np.zeros((6,136),dtype='float32'),'scales':np.ones(68),'shape':np.zeros(45),
                'object_rotation':np.tile(np.eye(3),(6,1,1)),'object_translation':np.ones((6,3)),
                'object_scale':np.array(1.),'report':np.array('preserved')}

    def test_only_selected_arrays_and_pinned_frame_may_change(self):
        c={k:v.copy() for k,v in self.p.items()};c['pose'][4,5]=.1;c['object_translation'][3,0]+=.01
        validate_pair(self.p,c,2,{0,4})
        for key,index in [('object_scale',()),('scales',0),('shape',0),('pose',(2,0))]:
            bad={k:v.copy() for k,v in c.items()};bad[key][index]+=1
            with self.assertRaises(ValueError):validate_pair(self.p,bad,2,{0,4})

    def test_missing_fields_nonfinite_and_untouched_metadata_rejected(self):
        for mutate in [lambda c:c.pop('report'),lambda c:c['pose'].__setitem__((4,0),np.nan),lambda c:c.__setitem__('report',np.array('changed'))]:
            c={k:v.copy() for k,v in self.p.items()};mutate(c)
            with self.assertRaises(ValueError):validate_pair(self.p,c,2,{0,4})

    def test_packed_alignment_rows_and_row_identity_are_checked_against_parquet(self):
        import pandas as pd
        rows=['t1_000000_000002_0_000000','t1_000000_000003_0_000000','t1_000000_000003_4_000000','t1_000000_999999_5_000000']
        a=pd.DataFrame(np.zeros((4,3)),index=rows,columns=['x','y','z'])
        b=a.copy();b.iloc[1,0]=1
        changed,_=validate_packed_rows(a,b.iloc[::-1],{0,4},{'0':2})
        self.assertEqual(changed.sum(),1)
        for index in [0,3]:
            bad=a.copy();bad.iloc[index,0]=1
            with self.assertRaises(ValueError):validate_packed_rows(a,bad,{0,4},{'0':2})
        with self.assertRaises(ValueError):validate_packed_rows(a,b.iloc[:3],{0,4},{'0':2})


if __name__=='__main__':unittest.main()
