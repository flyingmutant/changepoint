// Copyright 2020 Gregory Petrosyan <gregory.petrosyan@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package changepoint_test

import (
	"fmt"
	"reflect"
	"testing"

	"pgregory.net/changepoint"
)

var (
	big = []float64{
		8.51775781386248, 1013.74746867838, 969.560811828691, 9.99903626421251,
		11.3498239348095, 990.192098938082, 10.8362538563524, 11.0497172665872,
		1054.23425067251, 9.66601964023216, 1037.81637817117, 8.66558370477946,
		919.374881701026, 1131.1541993554, 11.2022838597658, 7.98843446730286,
		8.02589313889234, 10.8204269900969, 10.3681600526862, 8.93960790455779,
		9.35305979981456, 915.395965153361, 8.16535865503972, 10.2120572491007,
		1066.64002362996, 11.069835770419, 9.31344240206391, 10.2348674145244,
		11.6801773710108, 7.84243526689435, 9.21956447597874, 1112.57213641362,
		10.6780602207311, 1079.1784141977, 8.88378695422417, 8.70650073605106,
		8.77123401566279, 10.1077223378988, 9.38105207657018, 12.1034309729932,
		1053.30004708716, 955.498653876947, 953.97646035295, 9.80143691452122,
		987.671577491428, 8.9168418081165, 9.68107569533077, 10.0424589038011,
		9.63713378148637, 932.511669228627, 916.740706556999, 972.101180041745,
		1057.81026627134, 888.449133758924, 10.4439525673558, 9.91175069231749,
		9.16063280158654, 1094.72129249188, 983.216157217079, 878.618768427093,
		959.360944583076, 9.81400390730057, 1042.77719758456, 10.0499199152934,
		8.87431073340845, 9.76856626664346, 9.54692832839134, 11.3173847481645,
		10.8158661076597, 10.1762447729252, 8.59347190198671, 9.89783946138358,
		10.4458247651239, 10.5958375090298, 1048.89026506509, 9.53689245159815,
		10.3556711294034, 10.5111206253409, 10.9498649241074, 10.0522571754171,
		10.427691547751, 1060.88753031917, 9.69076437597138, 11.724350932454,
		10.9261779098129, 9.1663020020811, 11.7754734440436, 9.21696404189964,
		1109.00024831966, 8.77237176835653, 11.5707985076869, 9.28229456839916,
		1059.70559777354, 11.3686880467374, 10.7804164367935, 10.6682087852642,
		9.81848935380367, 1117.56915403446, 10.6748380014858, 8.92628692366044,
		21.0443737235031, 1105.72241345132, 19.4477654250736, 17.7821802121474,
		19.4781999899639, 820.585116566889, 22.7905163513292, 1074.92528880068,
		1013.45423941534, 1033.78179165301, 21.7033767555293, 21.2010125600959,
		1155.11887755134, 1124.70778630606, 19.3635057382183, 20.8681411244392,
		1025.46927962978, 20.9228886069588, 19.7414810417396, 17.6077812642339,
		18.7826030943573, 18.4230008183603, 21.5574158030065, 20.3540972663881,
		19.8951972303914, 20.5874066514834, 19.5436655021679, 871.540648869152,
		1131.09301822079, 909.512157959961, 21.1625922742, 20.1996974368549,
		21.4371296125597, 20.6326908527947, 19.003929460121, 22.1891456118526,
		846.131237958441, 21.1737225623976, 17.1273840427487, 20.6505297811944,
		21.8058704670258, 974.533374319673, 1038.80781759386, 916.525796601127,
		19.5550401051096, 23.1690470680813, 21.6574852061953, 23.6718803027189,
		1322.22337184262, 15.7593232457946, 1048.75496002333, 19.8561784114712,
		20.1016645168086, 21.9293783943216, 1227.01980787618, 1006.39031466135,
		20.8804270869218, 1105.5032121435, 18.7780711547771, 19.1795343556248,
		20.442492618286, 22.4707697720047, 1078.35810096477, 833.855347209093,
		20.3579023933224, 1046.39353756804, 969.025272754239, 18.265175253187,
		22.3140634142563, 912.906077936653, 879.367205674359, 21.6433957676533,
		20.9004212873356, 932.700450733285, 1022.41607841167, 16.3513165110277,
		19.6568173805112, 20.9655767366488, 21.9780598222067, 881.348203812156,
		18.2930220268125, 18.9998486812023, 22.3244524568253, 20.4548445140534,
		18.720101811033, 20.0178080242701, 20.2804893326285, 1019.0048151298,
		21.3107894997328, 21.3910419104093, 18.3314544873099, 18.4213348686018,
		16.7293099646236, 22.4272411724304, 18.8659567857605, 19.9033226277375,
		15.8084704011053, 23.5263915426487, 22.0073090762765, 21.5468281669676,
	}
)

func compareChangepoints(t *testing.T, data []float64, minSegment int, ref []int) {
	t.Helper()

	p := changepoint.NonParametric(data, minSegment)
	if !reflect.DeepEqual(p, ref) {
		t.Fatalf("got %#v instead of %#v", p, ref)
	}
}

func TestNonParametricConst(t *testing.T) {
	testData := []struct {
		name       string
		data       []float64
		minSegment int
		ref        []int
	}{
		{"Empty", []float64{}, 1, nil},
		{"Test1", []float64{3240, 3207, 2029, 3028, 3021, 2624, 3290, 2823, 3573}, 1, nil},
		{"Test2", big, 1, []int{99}},
		{"Test3", []float64{0, 0, 0, 0, 0, 100, 100, 100, 100}, 1, []int{4}},
		{"Test4", []float64{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2}, 1, []int{5, 11}},
		{"MinSegmentTooBig", []float64{0, 0, 0, 0, 1, 1, 1}, 4, nil},
		{"MinSegmentJustRight", []float64{0, 0, 0, 0, 1, 1, 1, 1}, 4, []int{3}},
	}

	for _, td := range testData {
		t.Run(td.name, func(t *testing.T) {
			compareChangepoints(t, td.data, td.minSegment, td.ref)
		})
	}
}

func TestNonParametricSeq(t *testing.T) {
	testData := []struct {
		start int
		end   int
		ref   []int
	}{
		{1, 1000, []int{15, 47, 79, 129, 204, 306, 432, 565, 691, 793, 868, 918, 950, 982}},
		{1, 10000, []int{26, 79, 136, 230, 387, 643, 1051, 1671, 2552, 3692, 4998, 6305, 7445, 8326, 8946, 9354, 9610, 9767, 9861, 9918, 9971}},
	}

	for _, td := range testData {
		t.Run(fmt.Sprintf("%v-%v", td.start, td.end), func(t *testing.T) {
			data := make([]float64, td.end-td.start+1)
			cur := float64(td.start)
			for i := range data {
				data[i] = cur
				cur++
			}

			compareChangepoints(t, data, 1, td.ref)
		})
	}
}
